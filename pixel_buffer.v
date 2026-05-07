// =============================================================================
// pixel_buffer.v  -  UART Pixel Buffer (784 bytes ? inference trigger)
// =============================================================================
//
// Receives exactly NUM_PIXELS bytes from the UART receiver and stores them
// in a dual-port block RAM. When the last byte arrives it asserts pixels_ready
// for exactly one clock cycle, which connects directly to snn_inference_live's
// start port (which must be in S_IDLE to catch it).
//
// READ LATENCY NOTE:
//   The read port is registered (synchronous BRAM output).
//   rd_data reflects the rd_addr presented in the PREVIOUS clock cycle.
//   snn_inference_live handles this via its S_PREFETCH state and careful
//   rd_addr management - do NOT replace it with snn_inference_v2 which
//   expects a zero-latency combinational read.
//
// Protocol:
//   1. Host sends 0xAB (SYNC_BYTE) - resets write pointer, starts new frame
//   2. Host sends 784 pixel bytes in raster order (row-major, top-left first)
//   3. pixels_ready pulses high for 1 cycle on the 784th byte
//   4. snn_inference_live must be in S_IDLE to catch this pulse
//
// Ports:
//   clk           - System clock
//   rst_n         - Active-low asynchronous reset
//   rx_valid      - From uart_rx: 1-cycle pulse when byte ready
//   rx_data       - From uart_rx: received byte
//   pixels_ready  - 1-cycle pulse when 784th byte is stored
//   rd_addr       - [9:0] Read address driven by snn_inference_live
//   rd_data       - [7:0] Registered pixel output (1-cycle latency)
//   pixel_count   - [9:0] Debug: bytes received in current frame
// =============================================================================
`timescale 1ns / 1ps

module pixel_buffer #(
    parameter NUM_PIXELS = 784,
    parameter SYNC_BYTE  = 8'hAB
)(
    input  wire        clk,
    input  wire        rst_n,
    // From uart_rx
    input  wire        rx_valid,
    input  wire [7:0]  rx_data,
    // To snn_inference_live
    output reg         pixels_ready,
    input  wire [9:0]  rd_addr,
    output wire [7:0]  rd_data,
    // Debug
    output wire [9:0]  pixel_count
);

    // -------------------------------------------------------------------------
    // Dual-port RAM - 784 × 8 bits
    // Port A: synchronous write  (uart_rx fills pixels as they arrive)
    // Port B: synchronous read   (snn_inference_live reads via rd_addr)
    //
    // (* ram_style = "block" *) forces BRAM36 inference.
    // Without this attribute Vivado chose LUT-RAM consuming ~2,251 LUTs.
    // With it, pixel_ram maps to 1 BRAM36 tile (784x8 = 6,272 bits < 36 Kb)
    // freeing those LUTs and reducing routing congestion.
    //
    // Both ports are synchronous so BRAM inference conditions are met:
    //   Write : pixel_ram[wr_ptr] <= rx_data       (posedge clk block)
    //   Read  : rd_data_r <= pixel_ram[rd_addr]    (posedge clk block)
    // 1-cycle read latency is unchanged - snn_inference_live already
    // handles it via S_PREFETCH and rd_addr prefetching.
    // -------------------------------------------------------------------------
    (* ram_style = "block" *) reg [7:0] pixel_ram [0:NUM_PIXELS-1];

    // Write pointer
    reg [9:0] wr_ptr;

    // Registered read port - 1-cycle latency
    reg [7:0] rd_data_r;
    always @(posedge clk) begin
        rd_data_r <= pixel_ram[rd_addr];
    end
    assign rd_data     = rd_data_r;
    assign pixel_count = wr_ptr;

    // -------------------------------------------------------------------------
    // Write logic + pixels_ready generation
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr       <= 10'd0;
            pixels_ready <= 1'b0;
        end else begin
            pixels_ready <= 1'b0;   // default: deassert every cycle

            if (rx_valid) begin
                if (rx_data == SYNC_BYTE) begin
                    // Sync byte - reset write pointer for a fresh frame.
                    // Any partially-received frame is discarded.
                    wr_ptr <= 10'd0;
                end else if (wr_ptr < NUM_PIXELS) begin
                    pixel_ram[wr_ptr] <= rx_data;
                    wr_ptr            <= wr_ptr + 1;

                    // 784th byte stored - trigger inference on next cycle
                    if (wr_ptr == NUM_PIXELS - 1)
                        pixels_ready <= 1'b1;
                end
                // Bytes beyond 784 before next sync are silently dropped
            end
        end
    end

endmodule