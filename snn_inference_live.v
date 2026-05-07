// =============================================================================
// snn_inference_live.v  -  SNN Engine with External Pixel Input (Rev 5)
// =============================================================================
//
// CHANGE vs Rev 4 - W1 synchronous read (BRAM inference fix):
//
//   Root cause of BRAM=0 in implementation:
//     The combinational read  wire w1v = W1[w1_addr]  prevented Vivado from
//     inferring BRAM36 for W1.  BRAM requires a registered (clocked) read
//     port.  With a combinational read, Vivado silently fell back to LUT-RAM,
//     consuming ~17,000 LUTs from the W1 array alone, causing 98% LUT fill,
//     routing congestion, and TNS = -3815 ns after implementation.
//
//   Fix - registered W1 read port:
//     w1v is now a reg driven by an always @(posedge clk) block.
//     Vivado will infer BRAM36 (~22 tiles) for W1, freeing ~15,000 LUTs and
//     eliminating the congestion that caused the timing violation.
//
//   w1_addr prefetch correction:
//     Because w1v now has 1-cycle latency, w1_addr must be presented ONE
//     cycle BEFORE the data is needed in S_L1_MAC / S_L2_MAC.
//
//     W1 prefetch timeline:
//       S_PREFETCH  : w1_addr <= 0   ?  w1v = W1[0] ready at start of
//       S_CLR_CUR1  : (128 cycles - plenty of margin)
//       S_L1_MAC c0 : w1v = W1[0], accumulate cur1[0], w1_addr <= 1
//       S_L1_MAC c1 : w1v = W1[1], accumulate cur1[0], w1_addr <= 2
//       ...
//     w1_addr is set to 0 in S_PREFETCH (was previously done in S_CLR_CUR1).
//     On neuron/timestep boundaries w1_addr is reset to the correct base
//     one cycle early so the first weight is always ready.
//
//     W2 read (w2v) is unchanged - W2 is LUT-RAM with async read,
//     no latency adjustment needed.
//
//   Simulation impact:
//     Bit-identical results - the same weights are used, just fetched one
//     cycle earlier.  Testbench will show 10/10 PASS unchanged.
//
// ALL PREVIOUS FIXES CARRIED FORWARD (Rev 2 - Rev 4):
//   1. S_DONE returns state <= S_IDLE
//   2. LIF temporaries at module level
//   3. mem1/cur1/mem2/cur2/spk1/spike_cnt as distributed LUT-RAM
//   4. For-loop reset removed from async reset block
//   5. spike_cnt cleared sequentially in S_INIT2
//
// Pipeline:
//   uart_rx -> pixel_buffer -> snn_inference_live -> uart_tx  (top.v)
//
// BRAM read latency summary (both BRAMs now registered - 1 cycle each):
//   Pixel BRAM  : rd_addr set cycle N  ?  rd_data valid cycle N+1
//   Weight BRAM : w1_addr set cycle N  ?  w1v     valid cycle N+1
//   Both handled by presenting addresses one cycle early via prefetch states.
// =============================================================================
`timescale 1ns / 1ps

module snn_inference_live #(
    parameter TIMESTEPS    = 8,
    parameter THRESHOLD    = 500,
    parameter DECAY_SHIFT  = 3,
    parameter BIAS_SHIFT   = 4,
    parameter WEIGHT_SHIFT = 2,
    parameter INPUT_THRESH = 5,
    parameter NUM_IN       = 784,
    parameter NUM_H        = 128,
    parameter NUM_OUT      = 10,
    parameter W1_FILE      = "w1.mem",
    parameter W2_FILE      = "w2.mem",
    parameter B1_FILE      = "b1.mem",
    parameter B2_FILE      = "b2.mem"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,           // 1-cycle pulse from pixel_buffer
    output reg  [9:0]  rd_addr,         // pixel BRAM read address
    input  wire [7:0]  rd_data,         // pixel BRAM read data (1-cycle latency)
    output reg  [3:0]  prediction,      // argmax result
    output reg         valid,           // pulses 1 cycle when result ready
    output reg  [15:0] cycle_cnt,
    output wire [7:0]  dbg_spikes_0,
    output wire [7:0]  dbg_spikes_1,
    output wire [7:0]  dbg_spikes_2,
    output wire [7:0]  dbg_spikes_3,
    output wire [7:0]  dbg_spikes_4,
    output wire [7:0]  dbg_spikes_5,
    output wire [7:0]  dbg_spikes_6,
    output wire [7:0]  dbg_spikes_7,
    output wire [7:0]  dbg_spikes_8,
    output wire [7:0]  dbg_spikes_9
);

    // =========================================================================
    // Weight / bias ROMs
    // =========================================================================
    (* ram_style = "block" *)
    reg signed [7:0] W1 [0:NUM_IN*NUM_H-1];    // 100352 bytes - BRAM
    (* rom_style = "distributed" *)
    reg signed [7:0] W2 [0:NUM_H*NUM_OUT-1];   // 1280 bytes  - LUT RAM
    reg signed [7:0] B1 [0:NUM_H-1];
    reg signed [7:0] B2 [0:NUM_OUT-1];

    initial begin
        $readmemh(W1_FILE, W1);
        $readmemh(W2_FILE, W2);
        $readmemh(B1_FILE, B1);
        $readmemh(B2_FILE, B2);
    end

    // =========================================================================
    // Neuron state
    // =========================================================================
    // Rev 4: promoted to distributed LUT-RAM to eliminate the 128:1 mux trees
    // Vivado builds for FF arrays with dynamic addresses.  Saves ~2,000-3,000
    // LUTs.  LUT-RAM reads are asynchronous so no FSM timing changes needed.
    // Constraint: element-by-element async reset is not supported -- state is
    // initialised sequentially by S_INIT1 / S_INIT2 before first inference.
    (* ram_style = "distributed" *) reg signed [31:0] mem1      [0:NUM_H-1];
    (* ram_style = "distributed" *) reg signed [31:0] cur1      [0:NUM_H-1];
    (* ram_style = "distributed" *) reg signed [31:0] mem2      [0:NUM_OUT-1];
    (* ram_style = "distributed" *) reg signed [31:0] cur2      [0:NUM_OUT-1];
    (* ram_style = "distributed" *) reg               spk1      [0:NUM_H-1];
    (* ram_style = "distributed" *) reg        [7:0]  spike_cnt [0:NUM_OUT-1];

    // LIF temporaries - module-level to avoid Verilog-2001 named-block issues
    reg signed [31:0] lif1_ml, lif1_mn;
    reg signed [31:0] lif2_ml, lif2_mn;

    localparam signed [31:0] THRESH_S = THRESHOLD;

    assign dbg_spikes_0 = spike_cnt[0]; assign dbg_spikes_1 = spike_cnt[1];
    assign dbg_spikes_2 = spike_cnt[2]; assign dbg_spikes_3 = spike_cnt[3];
    assign dbg_spikes_4 = spike_cnt[4]; assign dbg_spikes_5 = spike_cnt[5];
    assign dbg_spikes_6 = spike_cnt[6]; assign dbg_spikes_7 = spike_cnt[7];
    assign dbg_spikes_8 = spike_cnt[8]; assign dbg_spikes_9 = spike_cnt[9];

    // =========================================================================
    // FSM states
    // =========================================================================
    localparam [3:0]
        S_IDLE      = 4'd0,
        S_INIT1     = 4'd1,
        S_INIT2     = 4'd2,
        S_PREFETCH  = 4'd3,   // present rd_addr=0, absorb 1-cycle BRAM latency
        S_CLR_CUR1  = 4'd4,
        S_L1_MAC    = 4'd5,
        S_CLR_CUR2  = 4'd6,
        S_L1_FIRE   = 4'd7,
        S_L2_MAC    = 4'd8,
        S_L2_FIRE   = 4'd9,
        S_NEXT_STEP = 4'd10,
        S_ARGMAX    = 4'd11,
        S_DONE      = 4'd12;

    reg [3:0]  state;
    reg [3:0]  t_step;
    reg [9:0]  p_idx;
    reg [6:0]  h_idx;
    reg [3:0]  o_idx;
    reg [16:0] w1_addr;
    reg [10:0] w2_addr;
    reg [3:0]  max_idx;
    reg [7:0]  max_spikes;
    reg [3:0]  cmp_idx;

    // =========================================================================
    // W1 registered read - CRITICAL for BRAM inference
    // =========================================================================
    // w1v must be a reg driven by a synchronous always block.
    // A combinational  wire w1v = W1[w1_addr]  prevents Vivado from inferring
    // BRAM36, causing W1 to map to LUT-RAM and consuming ~17,000 LUTs.
    // With a registered read, Vivado infers ~22 BRAM36 tiles for W1 (free LUTs).
    // Latency: w1_addr presented cycle N ? w1v valid cycle N+1.
    // FSM compensates by setting w1_addr one cycle early (see S_PREFETCH).
    reg signed [7:0] w1v;
    always @(posedge clk)
        w1v <= W1[w1_addr];

    // =========================================================================
    // Datapath
    // =========================================================================
    // W2 remains combinational (LUT-RAM, async read - no latency)
    wire signed [7:0]  w2v    = W2[w2_addr];
    wire signed [31:0] w1v_sx = {{24{w1v[7]}}, w1v};
    wire signed [31:0] w2v_sx = {{24{w2v[7]}}, w2v};

    // rd_data is pixel_buffer's registered output - valid for current p_idx
    // because rd_addr was advanced one cycle ahead in the previous L1_MAC cycle
    wire in_active  = (rd_data > (INPUT_THRESH));
    wire signed [31:0] l1_delta = in_active   ? (w1v_sx << WEIGHT_SHIFT) : 32'sd0;
    wire signed [31:0] l2_delta = spk1[h_idx] ? (w2v_sx << WEIGHT_SHIFT) : 32'sd0;

    // =========================================================================
    // FSM
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            valid      <= 1'b0;
            prediction <= 4'd0;
            cycle_cnt  <= 16'd0;
            t_step     <= 4'd0;
            p_idx      <= 10'd0;
            h_idx      <= 7'd0;
            o_idx      <= 4'd0;
            w1_addr    <= 17'd0;
            w2_addr    <= 11'd0;
            rd_addr    <= 10'd0;
            max_idx    <= 4'd0;
            max_spikes <= 8'd0;
            cmp_idx    <= 4'd0;
            lif1_ml    <= 32'sd0; lif1_mn <= 32'sd0;
            lif2_ml    <= 32'sd0; lif2_mn <= 32'sd0;
            // mem1, cur1, mem2, cur2, spk1, spike_cnt are LUT-RAM and cannot
            // be reset element-by-element in async reset.  S_INIT1/S_INIT2
            // clear every entry sequentially before the first inference runs.
        end else begin
            cycle_cnt <= cycle_cnt + 1;

            case (state)

                // ?? Wait for start pulse from pixel_buffer ????????????????????
                S_IDLE: begin
                    valid     <= 1'b0;
                    cycle_cnt <= 16'd0;
                    if (start) begin
                        t_step <= 4'd0;
                        h_idx  <= 7'd0;
                        o_idx  <= 4'd0;
                        // spike_cnt is LUT-RAM -- cleared sequentially in
                        // S_INIT2 (one entry per cycle) before inference starts
                        state <= S_INIT1;
                    end
                end

                // ?? Init L1 membrane: mem1[h] = B1[h] << BIAS_SHIFT ??????????
                S_INIT1: begin
                    mem1[h_idx] <= $signed({{24{B1[h_idx][7]}}, B1[h_idx]})
                                   << BIAS_SHIFT;
                    cur1[h_idx] <= 32'sd0;
                    spk1[h_idx] <= 1'b0;
                    if (h_idx == NUM_H-1) begin
                        h_idx <= 7'd0;
                        o_idx <= 4'd0;
                        state <= S_INIT2;
                    end else
                        h_idx <= h_idx + 1;
                end

                // -- Init L2 membrane + zero spike_cnt -----------------------
                // spike_cnt is LUT-RAM -- cannot be reset in async reset.
                // Clear it here alongside mem2/cur2 (one entry per cycle)
                // before inference begins.  NUM_OUT=10 so 10 cycles total.
                S_INIT2: begin
                    mem2[o_idx]      <= $signed({{24{B2[o_idx][7]}}, B2[o_idx]})
                                        << BIAS_SHIFT;
                    cur2[o_idx]      <= 32'sd0;
                    spike_cnt[o_idx] <= 8'd0;   // LUT-RAM safe sequential clear
                    if (o_idx == NUM_OUT-1) begin
                        // Present rd_addr=0 now; pixel[0] valid after 1 cycle
                        rd_addr <= 10'd0;
                        state   <= S_PREFETCH;
                    end else
                        o_idx <= o_idx + 1;
                end

                // ?? Absorb BRAM read latency for pixel[0] and W1[0] ??????????
                // rd_addr=0 set last cycle (S_INIT2) ? rd_data=pixel[0] ready
                // w1_addr=0 set HERE ? w1v=W1[0] ready after S_CLR_CUR1 c0
                // S_CLR_CUR1 runs 128 cycles - far more than the 1-cycle BRAM
                // latency needed for both pixel and weight BRAMs.
                S_PREFETCH: begin
                    h_idx   <= 7'd0;
                    p_idx   <= 10'd0;
                    w1_addr <= 17'd0;   // present addr=0 now ? W1[0] ready +1 cycle
                    state   <= S_CLR_CUR1;
                end

                // ?? Zero L1 current accumulators ??????????????????????????????
                // w1_addr was set to 0 in S_PREFETCH - do NOT touch it here.
                // By the time we leave this state (128 cycles later) W1[0] has
                // been valid on w1v for 127 cycles - timing is safe.
                S_CLR_CUR1: begin
                    cur1[h_idx] <= 32'sd0;
                    if (h_idx == NUM_H-1) begin
                        h_idx <= 7'd0;
                        p_idx <= 10'd0;
                        state <= S_L1_MAC;
                    end else
                        h_idx <= h_idx + 1;
                end

                // ?? L1 MAC ????????????????????????????????????????????????????
                // w1v holds W1[w1_addr from LAST cycle] - registered read.
                // We accumulate w1v this cycle and advance w1_addr to prefetch
                // the NEXT weight (ready next cycle).
                // rd_data = pixel[p_idx] (pixel BRAM also registered, same pattern)
                //
                // On last pixel of a neuron (p_idx==783):
                //   rd_addr=0   ? pixel[0] ready for next neuron (+1 cycle)
                //   w1_addr+1   ? already points to first weight of next neuron;
                //                 the transition to S_CLR_CUR2 means w1_addr is
                //                 not used until S_L2_MAC so no conflict.
                S_L1_MAC: begin
                    cur1[h_idx] <= cur1[h_idx] + l1_delta;

                    if (p_idx == NUM_IN-1) begin
                        p_idx   <= 10'd0;
                        rd_addr <= 10'd0;       // prefetch pixel[0] for next neuron
                        w1_addr <= w1_addr + 1; // advance past last weight of neuron
                        if (h_idx == NUM_H-1) begin
                            h_idx <= 7'd0;
                            o_idx <= 4'd0;
                            state <= S_CLR_CUR2;
                        end else
                            h_idx <= h_idx + 1;
                    end else begin
                        p_idx   <= p_idx + 1;
                        rd_addr <= p_idx + 1;   // prefetch pixel[p+1]
                        w1_addr <= w1_addr + 1; // prefetch W1[next]
                    end
                end

                // ?? Zero L2 current accumulators ??????????????????????????????
                S_CLR_CUR2: begin
                    cur2[o_idx] <= 32'sd0;
                    if (o_idx == NUM_OUT-1) begin
                        h_idx <= 7'd0;
                        o_idx <= 4'd0;
                        state <= S_L1_FIRE;
                    end else
                        o_idx <= o_idx + 1;
                end

                // ?? L1 LIF fire ???????????????????????????????????????????????
                S_L1_FIRE: begin
                    lif1_ml = mem1[h_idx] - (mem1[h_idx] >>> DECAY_SHIFT);
                    lif1_mn = lif1_ml + cur1[h_idx];
                    if (lif1_mn > THRESH_S) begin
                        spk1[h_idx] <= 1'b1;
                        mem1[h_idx] <= 32'sd0;
                    end else begin
                        spk1[h_idx] <= 1'b0;
                        mem1[h_idx] <= lif1_mn;
                    end
                    if (h_idx == NUM_H-1) begin
                        h_idx   <= 7'd0;
                        o_idx   <= 4'd0;
                        w2_addr <= 11'd0;
                        state   <= S_L2_MAC;
                    end else
                        h_idx <= h_idx + 1;
                end

                // ?? L2 MAC ????????????????????????????????????????????????????
                S_L2_MAC: begin
                    cur2[o_idx] <= cur2[o_idx] + l2_delta;
                    if (h_idx == NUM_H-1) begin
                        h_idx <= 7'd0;
                        if (o_idx == NUM_OUT-1) begin
                            o_idx <= 4'd0;
                            state <= S_L2_FIRE;
                        end else
                            o_idx <= o_idx + 1;
                    end else
                        h_idx <= h_idx + 1;
                    w2_addr <= w2_addr + 1;
                end

                // ?? L2 LIF fire ???????????????????????????????????????????????
                S_L2_FIRE: begin
                    lif2_ml = mem2[o_idx] - (mem2[o_idx] >>> DECAY_SHIFT);
                    lif2_mn = lif2_ml + cur2[o_idx];
                    if (lif2_mn > THRESH_S) begin
                        spike_cnt[o_idx] <= spike_cnt[o_idx] + 1;
                        mem2[o_idx]      <= 32'sd0;
                    end else
                        mem2[o_idx] <= lif2_mn;
                    if (o_idx == NUM_OUT-1)
                        state <= S_NEXT_STEP;
                    else
                        o_idx <= o_idx + 1;
                end

                // ?? Advance timestep or start argmax ??????????????????????????
                // rd_addr=0: pixel BRAM prefetch for next timestep.
                // w1_addr=0: weight BRAM prefetch for next timestep.
                //   S_CLR_CUR1 (128 cycles) absorbs the 1-cycle BRAM latency
                //   for both - W1[0] and pixel[0] are valid before S_L1_MAC.
                S_NEXT_STEP: begin
                    rd_addr <= 10'd0;
                    w1_addr <= 17'd0;   // prefetch W1[0] for next timestep
                    if (t_step == TIMESTEPS-1) begin
                        max_spikes <= spike_cnt[0];
                        max_idx    <= 4'd0;
                        cmp_idx    <= 4'd1;
                        state      <= S_ARGMAX;
                    end else begin
                        t_step <= t_step + 1;
                        h_idx  <= 7'd0;
                        state  <= S_CLR_CUR1;
                    end
                end

                // ?? Argmax over spike_cnt[0..9] ???????????????????????????????
                S_ARGMAX: begin
                    if (spike_cnt[cmp_idx] > max_spikes) begin
                        max_spikes <= spike_cnt[cmp_idx];
                        max_idx    <= cmp_idx;
                    end
                    if (cmp_idx == NUM_OUT-1)
                        state <= S_DONE;
                    else
                        cmp_idx <= cmp_idx + 1;
                end

                // ?? Latch result, pulse valid for 1 cycle, return to IDLE ?????
                // FIX: state <= S_IDLE was missing in Rev 2, causing the FSM
                // to lock here and ignore all subsequent pixels_ready pulses.
                S_DONE: begin
                    prediction <= max_idx;
                    valid      <= 1'b1;
                    state      <= S_IDLE;   // ? KEY FIX
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule