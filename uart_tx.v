// =============================================================================
// uart_tx.v  —  8N1 UART Transmitter
// =============================================================================
//
// Transmits one byte with 8N1 framing at the configured baud rate.
// Asserts tx_busy while a transmission is in progress.
// A new byte is accepted only when tx_busy is low.
//
// Ports:
//   clk        — System clock
//   rst_n      — Active-low reset
//   tx_start   — Pulse high 1 cycle to begin transmitting tx_data
//   tx_data    — [7:0] Byte to send (sampled on tx_start rising edge)
//   tx         — UART TX line
//   tx_busy    — High while transmission is in progress
// =============================================================================
`timescale 1ns / 1ps

module uart_tx #(
    parameter CLKS_PER_BIT = 868   // 100 MHz / 115200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       tx_start,
    input  wire [7:0] tx_data,
    output reg        tx,
    output reg        tx_busy
);

    localparam [1:0]
        S_IDLE  = 2'd0,
        S_START = 2'd1,
        S_DATA  = 2'd2,
        S_STOP  = 2'd3;

    reg [1:0]  state;
    reg [15:0] clk_cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  tx_shift;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            tx       <= 1'b1;   // Idle line is high
            tx_busy  <= 1'b0;
            clk_cnt  <= 16'd0;
            bit_idx  <= 3'd0;
            tx_shift <= 8'd0;
        end else begin
            case (state)

                S_IDLE: begin
                    tx      <= 1'b1;
                    tx_busy <= 1'b0;
                    if (tx_start) begin
                        tx_shift <= tx_data;
                        tx_busy  <= 1'b1;
                        clk_cnt  <= 16'd0;
                        state    <= S_START;
                    end
                end

                S_START: begin
                    tx <= 1'b0;   // Start bit
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 16'd0;
                        bit_idx <= 3'd0;
                        state   <= S_DATA;
                    end else
                        clk_cnt <= clk_cnt + 1;
                end

                S_DATA: begin
                    tx <= tx_shift[bit_idx];  // LSB first
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 16'd0;
                        if (bit_idx == 3'd7) begin
                            bit_idx <= 3'd0;
                            state   <= S_STOP;
                        end else
                            bit_idx <= bit_idx + 1;
                    end else
                        clk_cnt <= clk_cnt + 1;
                end

                S_STOP: begin
                    tx <= 1'b1;   // Stop bit
                    if (clk_cnt == CLKS_PER_BIT - 1) begin
                        clk_cnt <= 16'd0;
                        state   <= S_IDLE;
                        tx_busy <= 1'b0;
                    end else
                        clk_cnt <= clk_cnt + 1;
                end

                default: begin
                    tx    <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
