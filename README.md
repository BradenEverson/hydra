# hydra 🐍

### A clone of the [triton](https://github.com/BradenEverson/hydra) crate, with cuda bindings for GPU runtime of matrix multiplication and backprop


## Installation

Use the package manager [cargo](https://crates.io/) to add [triton](https://crates.io/crates/triton_grow) to your rust project.

```bash
cargo add triton_hydra
```

or add the dependency directly in your **cargo.toml** file

```toml
[dependencies]
triton_hydra = "{version}"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```
## Usage

Triton acts as a typical neural network implementation, but allows for a more dynamic way of solving problems you may not know how to solve. Acting as a 'brute force' approach to the world of deep learning, after ```n``` epochs in the training process triton will evaluate the specific error of each neuron and column, deciding whether to add a neuron to a column, add a new column entirely, remove a neuron or remove a column. 

Triton will train and grow a desirable neural network until a specific accuracy is matched, returning the finished model

```rust
use triton_grow::network::{network::Network, activations, modes::Mode};

fn main() {
    let mut inputs = vec![vec![0.0,0.0],vec![1.0,0.0],vec![0.0,1.0],vec![1.0,1.0]];
    let mut outputs = vec![vec![0.0],vec![1.0],vec![1.0],vec![0.0]];
    let mut new_net: Network = Network::new(vec![2,3,1], activations::SIGMOID, 0.1);
    
    new_net = new_net.train_to_loss(inputs, outputs, 0.001, 100000, Mode::Avg, 0.001, 3, 10);
    println!("1 and 0: {:?}", new_net.feed_forward(&vec![1.0,0.0])[0].round());
    println!("0 and 1: {:?}", new_net.feed_forward(&vec![0.0,1.0])[0].round());
    println!("1 and 1: {:?}", new_net.feed_forward(&vec![1.0,1.0])[0].round());
    println!("0 and 0: {:?}", new_net.feed_forward(&vec![0.0,0.0])[0].round());
    println!("Net network made: {:?}", new_net.layers);

}
```
## Proven Results

Upon testing Triton's self growth method against a traditional preconfigured network model. Three neural networks were all tasked with learning a simple **XOR predictor** with the following inputs and outputs:

### Inputs
```
[ 1.0 , 0.0 ]
[ 0.0 , 1.0 ]
[ 0.0 , 0.0 ]
[ 1.0 , 1.0 ]
```

### Outputs
```
[ 1.0 ]
[ 1.0 ]
[ 0.0 ]
[ 0.0 ]
```

### Testing

| Model Name    | Layers {input -[hidden] - output} | Epochs Needed to Get 0.001 Avg Loss |
| ------------- | ------------- | ------------- |
| Minimum  | 2 - { *3* } - 1  |  7,880,000 |
| Well Fit  | 2 - { *3 - 4 - 3* } - 1 | 2,790,000  |
| Triton  | 2 - { *self growing* } - 1 | 150,000  |

Triton was 98.09% more efficient than the minimum fit model, and 94.62% more than even the well fit model.


## TODO

Currently, triton is in a very beta stage, the following features are still in development:

 - [ ]  Mutating a neural network (1/2)
    - [X]  Adding a new layer with ```n``` neurons into any point of an existent network
    - [ ]  Removing a layer from an existent network
- [X]  Back propegation only affecting a single column (allows for a newly added layer to 'catch up')
- [X]  *Analysis* mode during back propegation allowing for all individual errors to be recorded
- [X]  Updated training function
    - [X]  Input desired success rate
    - [X]  Dynamic error analysis to allow for choosing if the network should grow or shrink
    - [X]  Acceptable threshold of +/- in the errors to allow for a less punishing learning process especially when a new neuron layer has been added
- [X]  Model serialization (serde)

## License

[MIT](https://choosealicense.com/licenses/mit/)

