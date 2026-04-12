# LSTM

- Designed for 1D sequence data /text)
- Uses a fully connected matrix for the input
- Treats input as 1D vector

LSTM is designed to solve the "memory problem" in sequential data, where 3 gates control information flow.
 - [[forget-gate]]: decides on discarding information
 - [[input-gate]]: decides which new information to store in the memory cell
 - [[output-gate]]: decides which parts of the memory cell state to output as hidden stade for next time step
 