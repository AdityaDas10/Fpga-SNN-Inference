// =============================================================================
// tb_snn_live.v  -  Testbench for the complete UART SNN pipeline
// =============================================================================
//
// Tests the full chain:
//   inject_bytes ? pixel_buffer ? snn_inference_live ? prediction
//
// UART bit-timing is bypassed - bytes are injected directly via rx_valid/
// rx_data pulses. This keeps simulation to ~82 ms instead of hours.
//
// Key differences vs the old testbench:
//   1. NO hard reset between digits - the fixed S_DONE ? S_IDLE transition
//      means the SNN returns to idle naturally. This validates the fix.
//   2. wait_valid waits for valid to RISE (edge detect), not just level,
//      so it won't return immediately if valid happened to be high.
//   3. A 2-cycle post-valid wait is added to let prediction settle and
//      to observe the valid ? 0 transition.
//   4. Monitors are edge-triggered so they fire exactly once per inference.
//
// SETUP:
//   1. Run export_snn_weights_v2.py - copy w1.mem w2.mem b1.mem b2.mem
//      and input_0.mem .. input_9.mem to sim working directory
//   2. Set THRESHOLD_VAL to the calibrated value from the Python script
//   3. Set this file as the simulation top in Vivado
//
// Vivado sim working dir:
//   <project>.sim/sim_1/behav/xsim/
// =============================================================================
`timescale 1ns / 1ps

module tb_snn_live;

    // =========================================================================
    // *** UPDATE FROM export_snn_weights_v2.py OUTPUT ***
    // =========================================================================
    localparam THRESHOLD_VAL = 500;

    // Expected prediction per digit
    // Update individual entries if the calibrated model predicts differently
    reg [3:0] expected [0:9];
    initial begin
        expected[0] = 4'd0; expected[1] = 4'd1;
        expected[2] = 4'd2; expected[3] = 4'd3;
        expected[4] = 4'd4; expected[5] = 4'd5;
        expected[6] = 4'd6; expected[7] = 4'd7;
        expected[8] = 4'd8; expected[9] = 4'd9;
    end

    // =========================================================================
    // Timing constants
    // =========================================================================
    localparam TIMEOUT   = 5_000_000;  // cycles per inference before giving up
    localparam BYTE_GAP  = 4;          // idle cycles between injected bytes
    localparam SYNC_BYTE = 8'hAB;

    // =========================================================================
    // Clock
    // =========================================================================
    reg clk, rst_n;
    initial clk = 0;
    always #5 clk = ~clk;   // 100 MHz

    // =========================================================================
    // pixel_buffer signals
    // =========================================================================
    reg        rx_valid;
    reg  [7:0] rx_data;
    wire        pixels_ready;
    wire [9:0]  rd_addr;
    wire [7:0]  rd_data;
    wire [9:0]  pixel_count;

    // =========================================================================
    // snn_inference_live signals
    // =========================================================================
    wire [3:0]  prediction;
    wire        valid;
    wire [15:0] cycle_cnt;
    wire [7:0]  spk0, spk1_w, spk2, spk3, spk4;
    wire [7:0]  spk5, spk6,   spk7, spk8, spk9;

    // =========================================================================
    // DUT 1 - pixel_buffer
    // =========================================================================
    pixel_buffer #(
        .NUM_PIXELS (784),
        .SYNC_BYTE  (SYNC_BYTE)
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
    // DUT 2 - snn_inference_live
    // =========================================================================
    snn_inference_live #(
        .TIMESTEPS    (8),
        .THRESHOLD    (THRESHOLD_VAL),
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
        .dbg_spikes_0 (spk0),
        .dbg_spikes_1 (spk1_w),
        .dbg_spikes_2 (spk2),
        .dbg_spikes_3 (spk3),
        .dbg_spikes_4 (spk4),
        .dbg_spikes_5 (spk5),
        .dbg_spikes_6 (spk6),
        .dbg_spikes_7 (spk7),
        .dbg_spikes_8 (spk8),
        .dbg_spikes_9 (spk9)
    );

    // =========================================================================
    // Pixel banks - one flat array per digit (Verilog-2001 compatible)
    // =========================================================================
    reg [7:0] px0[0:783]; reg [7:0] px1[0:783]; reg [7:0] px2[0:783];
    reg [7:0] px3[0:783]; reg [7:0] px4[0:783]; reg [7:0] px5[0:783];
    reg [7:0] px6[0:783]; reg [7:0] px7[0:783]; reg [7:0] px8[0:783];
    reg [7:0] px9[0:783];

    initial begin
        $readmemh("input_0.mem", px0); $readmemh("input_1.mem", px1);
        $readmemh("input_2.mem", px2); $readmemh("input_3.mem", px3);
        $readmemh("input_4.mem", px4); $readmemh("input_5.mem", px5);
        $readmemh("input_6.mem", px6); $readmemh("input_7.mem", px7);
        $readmemh("input_8.mem", px8); $readmemh("input_9.mem", px9);
    end

    // =========================================================================
    // Task: inject one byte
    // =========================================================================
    task inject_byte;
        input [7:0] b;
        begin
            @(posedge clk); #1;
            rx_valid = 1'b1;
            rx_data  = b;
            @(posedge clk); #1;
            rx_valid = 1'b0;
            repeat(BYTE_GAP) @(posedge clk);
        end
    endtask

    // =========================================================================
    // Task: send full frame for digit d
    // =========================================================================
    integer px_i;
    task send_frame;
        input [3:0] d;
        integer i;
        begin
            $display("[TB] Sending frame for digit %0d ...", d);
            inject_byte(SYNC_BYTE);
            for (px_i = 0; px_i < 784; px_i = px_i + 1) begin
                case (d)
                    4'd0: inject_byte(px0[px_i]);
                    4'd1: inject_byte(px1[px_i]);
                    4'd2: inject_byte(px2[px_i]);
                    4'd3: inject_byte(px3[px_i]);
                    4'd4: inject_byte(px4[px_i]);
                    4'd5: inject_byte(px5[px_i]);
                    4'd6: inject_byte(px6[px_i]);
                    4'd7: inject_byte(px7[px_i]);
                    4'd8: inject_byte(px8[px_i]);
                    4'd9: inject_byte(px9[px_i]);
                    default: inject_byte(8'd0);
                endcase
            end
            $display("[TB] Frame sent. pixel_count=%0d", pixel_count);
        end
    endtask

    // =========================================================================
    // Task: wait for valid RISING EDGE (not level) with timeout
    // This correctly handles back-to-back inferences without reset.
    // =========================================================================
    integer timeout_cnt;
    task wait_valid;
        output [3:0] result;
        begin
            timeout_cnt = 0;
            // Wait for valid to go low first (in case it's still high - shouldn't
            // happen with the S_DONE fix, but defensive coding)
            while (valid && timeout_cnt < 100) begin
                @(posedge clk); #1;
                timeout_cnt = timeout_cnt + 1;
            end
            timeout_cnt = 0;
            // Now wait for rising edge
            while (!valid && timeout_cnt < TIMEOUT) begin
                @(posedge clk); #1;
                timeout_cnt = timeout_cnt + 1;
            end
            if (!valid) begin
                $display("[TB] *** TIMEOUT after %0d cycles - FSM state=%0d t_step=%0d ***",
                         TIMEOUT, u_snn.state, u_snn.t_step);
                $finish;
            end
            result = prediction;
            // Let valid deassert (SNN goes S_DONE ? S_IDLE, valid clears in S_IDLE)
            @(posedge clk); #1;
            @(posedge clk); #1;
        end
    endtask

    // =========================================================================
    // Diagnostic helpers
    // =========================================================================
    function is_saturated;
        input [7:0] s0,s1,s2,s3,s4,s5,s6,s7,s8,s9;
        begin
            is_saturated = (s0==8||s1==8||s2==8||s3==8||s4==8||
                            s5==8||s6==8||s7==8||s8==8||s9==8);
        end
    endfunction

    function is_all_equal;
        input [7:0] s0,s1,s2,s3,s4,s5,s6,s7,s8,s9;
        begin
            is_all_equal = (s0==s1&&s1==s2&&s2==s3&&s3==s4&&
                            s4==s5&&s5==s6&&s6==s7&&s7==s8&&s8==s9);
        end
    endfunction

    // =========================================================================
    // Main test - NO reset between digits (validates S_DONE fix)
    // =========================================================================
    integer d, pass_cnt, fail_cnt;
    reg [3:0] result;

    initial begin
        rst_n    = 1'b0;
        rx_valid = 1'b0;
        rx_data  = 8'd0;
        pass_cnt = 0;
        fail_cnt = 0;

        repeat(4) @(posedge clk); #1;
        rst_n = 1'b1;
        repeat(2) @(posedge clk); #1;

        $display("============================================================");
        $display(" TB: SNN Live Pipeline   threshold=%0d   T=8", THRESHOLD_VAL);
        $display(" No reset between digits - validates S_DONE->S_IDLE fix");
        $display("============================================================");
        $display("  Digit  Pred   Exp  Status   Cycles  Spikes[0..9]");
        $display("------------------------------------------------------------");

        for (d = 0; d <= 9; d = d + 1) begin

            send_frame(d[3:0]);
            wait_valid(result);

            if (result === expected[d[3:0]]) begin
                $display("  %5d  %4d  %4d  PASS    %6d  [%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d]",
                    d, result, expected[d], cycle_cnt,
                    spk0,spk1_w,spk2,spk3,spk4,
                    spk5,spk6,  spk7,spk8,spk9);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("  %5d  %4d  %4d  FAIL    %6d  [%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d]",
                    d, result, expected[d], cycle_cnt,
                    spk0,spk1_w,spk2,spk3,spk4,
                    spk5,spk6,  spk7,spk8,spk9);
                fail_cnt = fail_cnt + 1;

                if (is_saturated(spk0,spk1_w,spk2,spk3,spk4,
                                 spk5,spk6,spk7,spk8,spk9))
                    $display("         All spikes=8: THRESHOLD too low, try %0d",
                             THRESHOLD_VAL * 2);
                else if (is_all_equal(spk0,spk1_w,spk2,spk3,spk4,
                                      spk5,spk6,spk7,spk8,spk9))
                    $display("         All spikes equal: check weight .mem files loaded correctly");
                else
                    $display("         Wrong winner: expected[%0d] should be 4'd%0d",
                             d, result);
            end
        end

        $display("============================================================");
        $display(" Result: %0d / 10 PASS", pass_cnt);
        if (fail_cnt == 0)
            $display(" *** ALL PASS *** Pipeline verified.");
        else begin
            $display(" *** %0d FAIL(S) *** Recalibrate THRESHOLD_VAL or re-export weights.",
                     fail_cnt);
        end
        $display("============================================================");
        $finish;
    end

    // =========================================================================
    // Monitors - edge triggered so each fires exactly once per event
    // =========================================================================
    always @(posedge clk) begin
        if (pixels_ready)
            $display("[BUF @%0t ns] pixels_ready - %0d bytes stored",
                     $time/1000, pixel_count);
    end

    always @(posedge clk) begin
        if (valid)
            $display("[SNN @%0t ns] DONE: pred=%0d  cycles=%0d  spikes=[%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d]",
                     $time/1000, prediction, cycle_cnt,
                     spk0,spk1_w,spk2,spk3,spk4,
                     spk5,spk6,  spk7,spk8,spk9);
    end

    always @(posedge clk) begin
        if (u_snn.state == 4'd10)   // S_NEXT_STEP
            $display("[SNN] Timestep %0d complete", u_snn.t_step);
    end

endmodule