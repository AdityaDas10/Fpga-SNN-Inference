// =============================================================================
// top_basys3_snn_live.v  -  SNN + UART Live Input Top-Level (Basys3)
// =============================================================================
//
// Combines the UART pixel receiver with the SNN inference engine.
// Send any 28x28 image from a PC and get the predicted digit back.
//
// Protocol (115200 8N1):
//   PC -> FPGA : 0xAB (sync byte) + 784 pixel bytes (row-major, uint8)
//   FPGA -> PC : 0xBB (marker) + 1 byte prediction (0x00-0x09)
//
// On-board USB-UART (FT2232HQ):
//   B18 - FPGA RX (receives from PC)
//   A18 - FPGA TX (sends to PC)
//
// No re-synthesis needed to test new images - just send from Python.
//
// LED indicators:
//   LED[15]   - valid (inference complete, pulses 1 cycle)
//   LED[14]   - blinks as UART bytes arrive
//   LED[13:12] - 00 (unused)
//   LED[11:8] - spike count of winning neuron (confidence indicator)
//   LED[7:4]  - 0000 (unused)
//   LED[3:0]  - predicted digit (holds until next inference)
//
// FIXES vs original:
//   1. valid/prediction are now held in registers (valid_latch, pred_latch)
//      so LED[3:0] and LED[15] stay stable until the next result arrives.
//      The raw valid from snn_inference_live is a 1-cycle pulse (after
//      the S_DONE fix) - without latching, LED[15] would be invisible.
//
//   2. RS_IDLE now uses valid_rise (edge detect on the latched valid) so
//      the result sender triggers exactly once per inference regardless of
//      how many cycles valid stays high.
//
//   3. RS_MARKER ? RS_RESULT transition waits for tx_busy to go HIGH first
//      (confirming the TX has actually started) before waiting for it to go
//      LOW again.  The original checked !tx_busy && !tx_start one cycle after
//      asserting tx_start, which is still !tx_busy (tx_busy goes high one
//      cycle after tx_start) - causing it to immediately re-fire.
//
//   4. winner_spikes mux and LED[11:8] now use pred_latch so they stay
//      stable between inferences.
// =============================================================================
`timescale 1ns / 1ps

module top_basys3_snn_live (
    input  wire        clk,       // W5  100 MHz
    input  wire        btnU,      // T18 reset (active high button ? active low rst_n)
    input  wire        uart_rxd,  // B18 UART RX from PC
    output wire        uart_txd,  // A18 UART TX to PC
    output wire [6:0]  seg,
    output wire        dp,
    output wire [3:0]  an,
    output wire [15:0] led
);

    wire rst_n = ~btnU;

    // =========================================================================
    // UART Receiver
    // =========================================================================
    wire       rx_valid;
    wire [7:0] rx_data;

    uart_rx #(.CLKS_PER_BIT(868)) u_rx (
        .clk        (clk),
        .rst_n      (rst_n),
        .rx         (uart_rxd),
        .data_valid (rx_valid),
        .data_out   (rx_data)
    );

    // =========================================================================
    // Pixel Buffer - 784-byte dual-port RAM
    // Fires pixels_ready after 784th byte, triggering SNN start
    // =========================================================================
    wire        pixels_ready;
    wire [9:0]  rd_addr;
    wire [7:0]  rd_data;
    wire [9:0]  pixel_count;

    pixel_buffer #(
        .NUM_PIXELS (784),
        .SYNC_BYTE  (8'hAB)
    ) u_pbuf (
        .clk          (clk),
        .rst_n        (rst_n),
        .rx_valid     (rx_valid),
        .rx_data      (rx_data),
        .pixels_ready (pixels_ready),
        .rd_addr      (rd_addr),
        .rd_data      (rd_data),
        .pixel_count  (pixel_count)
    );

    // =========================================================================
    // SNN Inference Engine - live pixel input version
    // =========================================================================
    wire [3:0]  prediction;
    wire        valid;          // 1-cycle pulse (snn_inference_live Rev 3)
    wire [15:0] cycle_cnt;
    wire [7:0]  dbg_spk_0, dbg_spk_1, dbg_spk_2, dbg_spk_3, dbg_spk_4;
    wire [7:0]  dbg_spk_5, dbg_spk_6, dbg_spk_7, dbg_spk_8, dbg_spk_9;

    snn_inference_live #(
        .TIMESTEPS    (8),
        .THRESHOLD    (500),       // UPDATE from export_snn_weights_v2.py
        .DECAY_SHIFT  (3),
        .BIAS_SHIFT   (4),
        .WEIGHT_SHIFT (2),
        .INPUT_THRESH (5),
        .NUM_IN       (784),
        .NUM_H        (128),
        .NUM_OUT      (10),
        .W1_FILE      ("w1.mem"),
        .W2_FILE      ("w2.mem"),
        .B1_FILE      ("b1.mem"),
        .B2_FILE      ("b2.mem")
    ) u_snn (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (pixels_ready),
        .rd_addr      (rd_addr),
        .rd_data      (rd_data),
        .prediction   (prediction),
        .valid        (valid),
        .cycle_cnt    (cycle_cnt),
        .dbg_spikes_0 (dbg_spk_0),
        .dbg_spikes_1 (dbg_spk_1),
        .dbg_spikes_2 (dbg_spk_2),
        .dbg_spikes_3 (dbg_spk_3),
        .dbg_spikes_4 (dbg_spk_4),
        .dbg_spikes_5 (dbg_spk_5),
        .dbg_spikes_6 (dbg_spk_6),
        .dbg_spikes_7 (dbg_spk_7),
        .dbg_spikes_8 (dbg_spk_8),
        .dbg_spikes_9 (dbg_spk_9)
    );

    // =========================================================================
    // FIX 1 - Latch valid and prediction so LEDs/display hold between frames
    // =========================================================================
    // valid from snn_inference_live is a 1-cycle pulse.
    // valid_latch stays high until the next inference begins (pixels_ready).
    // pred_latch holds the last prediction stably.
    reg        valid_latch;
    reg [3:0]  pred_latch;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_latch <= 1'b0;
            pred_latch  <= 4'd0;
        end else begin
            if (valid) begin
                valid_latch <= 1'b1;
                pred_latch  <= prediction;
            end else if (pixels_ready) begin
                // New frame starting - clear the "done" indicator
                valid_latch <= 1'b0;
            end
        end
    end

    // =========================================================================
    // FIX 2 - Result sender: transmit 0xBB + prediction over UART when done
    // =========================================================================
    // Uses valid (the raw 1-cycle pulse) as the trigger so it fires exactly
    // once per inference.  The 3-state FSM sends 0xBB then the digit byte.
    //
    // FIX 3 - RS_MARKER transition:
    //   Original bug: checked !tx_busy one cycle after asserting tx_start.
    //   tx_busy goes high one cycle AFTER tx_start, so the original code saw
    //   !tx_busy=1 and immediately jumped to RS_RESULT before the marker
    //   byte had even started transmitting, then fired the second tx_start
    //   right away - overwriting the marker with the result byte.
    //
    //   Fix: wait for tx_busy to assert (confirming TX started), then wait
    //   for it to deassert (confirming TX finished) before sending byte 2.
    // =========================================================================
    wire       tx_busy;
    reg        tx_start;
    reg [7:0]  tx_data;

    uart_tx #(.CLKS_PER_BIT(868)) u_tx (
        .clk      (clk),
        .rst_n    (rst_n),
        .tx_start (tx_start),
        .tx_data  (tx_data),
        .tx       (uart_txd),
        .tx_busy  (tx_busy)
    );

    localparam [1:0]
        RS_IDLE   = 2'd0,
        RS_MARKER = 2'd1,   // waiting for 0xBB to finish
        RS_RESULT = 2'd2;   // waiting for digit byte to finish

    reg [1:0] rs_state;
    reg       tx_busy_prev;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rs_state   <= RS_IDLE;
            tx_start   <= 1'b0;
            tx_data    <= 8'd0;
            tx_busy_prev <= 1'b0;
        end else begin
            tx_start     <= 1'b0;           // default: deassert
            tx_busy_prev <= tx_busy;

            case (rs_state)

                // Wait for a new valid pulse
                RS_IDLE: begin
                    if (valid && !tx_busy) begin
                        tx_data  <= 8'hBB;
                        tx_start <= 1'b1;
                        rs_state <= RS_MARKER;
                    end
                end

                // 0xBB is transmitting - wait for tx_busy to go low
                // (tx_busy goes high one cycle after tx_start, so we wait
                //  for the falling edge: tx_busy_prev=1 ? tx_busy=0)
                RS_MARKER: begin
                    if (tx_busy_prev && !tx_busy) begin
                        tx_data  <= {4'b0000, pred_latch};
                        tx_start <= 1'b1;
                        rs_state <= RS_RESULT;
                    end
                end

                // Digit byte is transmitting - wait for it to finish
                RS_RESULT: begin
                    if (tx_busy_prev && !tx_busy)
                        rs_state <= RS_IDLE;
                end

                default: rs_state <= RS_IDLE;
            endcase
        end
    end

    // =========================================================================
    // 7-Segment Display - shows predicted digit, blanks while inferring
    // =========================================================================
    seg7_ctrl u_seg7 (
        .clk       (clk),
        .rst_n     (rst_n),
        .digit_val (pred_latch),   // stable latched value
        .valid     (valid_latch),  // stays high between inferences
        .seg       (seg),
        .an        (an),
        .dp        (dp)
    );

    // =========================================================================
    // Spike confidence mux - shows winner neuron's spike count on LED[11:8]
    // Uses pred_latch so it holds stable between inferences
    // =========================================================================
    reg [7:0] winner_spikes;
    always @(*) begin
        case (pred_latch)
            4'd0: winner_spikes = dbg_spk_0;
            4'd1: winner_spikes = dbg_spk_1;
            4'd2: winner_spikes = dbg_spk_2;
            4'd3: winner_spikes = dbg_spk_3;
            4'd4: winner_spikes = dbg_spk_4;
            4'd5: winner_spikes = dbg_spk_5;
            4'd6: winner_spikes = dbg_spk_6;
            4'd7: winner_spikes = dbg_spk_7;
            4'd8: winner_spikes = dbg_spk_8;
            4'd9: winner_spikes = dbg_spk_9;
            default: winner_spikes = 8'd0;
        endcase
    end

    // =========================================================================
    // LEDs
    // =========================================================================
    assign led[15]   = valid_latch;             // stays lit after inference
    assign led[14]   = rx_valid;                // blinks on each received byte
    assign led[13:12] = 2'b00;
    assign led[11:8] = winner_spikes[3:0];      // confidence (max = 8 spikes)
    assign led[7:4]  = 4'b0000;
    assign led[3:0]  = valid_latch ? pred_latch : 4'b0000;

endmodule