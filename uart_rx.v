// =============================================================================
// uart_rx.v  —  8N1 UART Receiver
// =============================================================================
//
// Standard 8N1 framing: 1 start bit, 8 data bits, 1 stop bit, no parity.
// Oversampling at 16× for robust centre-sample timing.
//
// Baud rate is set by the CLKS_PER_BIT parameter:
//   CLKS_PER_BIT = f_clk / baud_rate
//   e.g. 100 MHz / 115200 = 868
//
// Ports:
//   clk          — System clock
//   rst_n        — Active-low reset
//   rx           — UART RX line (async input, double-registered internally)
//   data_valid   — Pulses high for 1 cycle when a full byte is received
//   data_out     — [7:0] Received byte, valid when data_valid=1
// =============================================================================
`timescale 1ns / 1ps

module uart_rx #(
    parameter CLKS_PER_BIT = 868   // 100 MHz / 115200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,
    output reg        data_valid,
    output reg  [7:0] data_out
);

    // -------------------------------------------------------------------------
    // Double-register rx to cross clock domains cleanly
    // -------------------------------------------------------------------------
    reg rx_d1, rx_d2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_d1 <= 1'b1;
            rx_d2 <= 1'b1;
        end else begin
            rx_d1 <= rx;
            rx_d2 <= rx_d1;
        end
    end

    // -------------------------------------------------------------------------
    // FSM states
    // -------------------------------------------------------------------------
    localparam [2:0]
        S_IDLE  = 3'd0,
        S_START = 3'd1,
        S_DATA  = 3'd2,
        S_STOP  = 3'd3;

    reg [2:0]  state;
    reg [15:0] clk_cnt;   // Baud rate counter
    reg [2:0]  bit_idx;   // Current data bit (0–7)
    reg [7:0]  rx_shift;  // Shift register

    // -------------------------------------------------------------------------
    // FSM
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            clk_cnt    <= 16'd0;
            bit_idx    <= 3'd0;
            rx_shift   <= 8'd0;
            data_valid <= 1'b0;
            data_out   <= 8'd0;
        end else begin
            data_valid <= 1'b0; // default: de-assert each cycle

            case (state)

                // Wait for falling edge on RX (start bit)
                S_IDLE: begin
                    if (!rx_d2) begin
                        clk_cnt <= 16'd0;
                        state   <= S_START;
                    end
                end

                // Wait to the middle of the start bit and verify it's still low
                S_START: begin
                    if (clk_cnt == (CLKS_PER_BIT / 2) - 1) begin
                        if (!rx_d2) begin
                            clk_cnt <= 16'd0;
                            bit_idx <= 3'd0;
                            state   <= S_DATA;
                        end else
                            state <= S_IDLE; // Noise — abort
                    end else
                        clk_cnt <= clk_cnt + 1;
                end

                // Sample each data bit at the centre of its bit period
                S_DATA: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt            <= 16'd0;
                        rx_shift[bit_idx]  <= rx_d2;  // LSB first

                        if (bit_idx == 3'd7) begin
                            bit_idx <= 3'd0;
                            state   <= S_STOP;
                        end else
                            bit_idx <= bit_idx + 1;
                    end else
                        clk_cnt <= clk_cnt + 1;
                end

                // Consume the stop bit then output the byte
                S_STOP: begin
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt    <= 16'd0;
                        data_valid <= 1'b1;
                        data_out   <= rx_shift;
                        state      <= S_IDLE;
                    end else
                        clk_cnt <= clk_cnt + 1;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
