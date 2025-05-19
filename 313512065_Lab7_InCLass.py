import torch
import matplotlib.pyplot as plt

# 1. Vocabulary
vocab = { "The": 0, "movie": 1, "is": 2, "good": 3, "bad": 4}
def word_to_onehot(word):
    vec = torch.zeros(len(vocab), 1)
    vec[vocab[word]] = 1.0
    return vec

# 2. Training Samples
train_data = [
    (["The", "movie", "is", "good"], 1),
    (["The", "movie", "is", "bad"], 0)
]

# 3. Define RNN Parameters
n = 2  # hidden size
m = len(vocab)  # vocab size
T = 4  # number of words

torch.manual_seed(42)

# Initialize Weights (n x n), (n x m), (n x 1), (2 x n), (2 x 1)
W = torch.randn(n, n, requires_grad=True)  # hidden to hidden
U = torch.randn(n, m, requires_grad=True)  # input to hidden
b = torch.randn(n, 1, requires_grad=True)  # bias for hidden state
W_out = torch.randn(2, n, requires_grad=True)  # hidden to output
b_out = torch.randn(2, 1, requires_grad=True)  # bias for output

# 4. Training Setup
learning_rate = 0.1
num_epochs = 300
loss_fn = torch.nn.CrossEntropyLoss()
loss_history = []

# 5. Training Loop
for epoch in range(num_epochs):
    total_loss = 0.0
    for sentence, label in train_data:
        inputs = [word_to_onehot(word) for word in sentence]
        s_prev = torch.zeros(n, 1)
        
        for x_t in inputs:
            # Compute s_t = tanh(W @ s_prev + U @ x_t + b)
            s_t = torch.tanh(W @ s_prev + U @ x_t + b)
            s_prev = s_t
        
        # Output layer: logits = W_out @ s_t + b_out
        logits = W_out @ s_t + b_out
        logits = logits.view(1, -1)

        # Loss
        target = torch.tensor([label])
        loss = loss_fn(logits, target)
        total_loss += loss.item()

        # Backward
        loss.backward()

        # Manual update (SGD)
        with torch.no_grad():
            W -= learning_rate * W.grad
            U -= learning_rate * U.grad
            b -= learning_rate * b.grad
            W_out -= learning_rate * W_out.grad
            b_out -= learning_rate * b_out.grad

            # Zero gradients after updating
            W.grad.zero_()
            U.grad.zero_()
            b.grad.zero_()
            W_out.grad.zero_()
            b_out.grad.zero_()
    
    loss_history.append(total_loss)
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss:.4f}")

# 6. Plot Loss
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

# 7. Test After Training
print("\n=== Testing ===")
test_sentences = [
    ["The", "movie", "is", "bad"],
    ["The", "movie", "is", "good"]
]

for test_sentence in test_sentences:
    inputs = [word_to_onehot(word) for word in test_sentence]
    s_prev = torch.zeros(n, 1)
    for x_t in inputs:
        s_t = torch.tanh(W @ s_prev + U @ x_t + b)
        s_prev = s_t
    logits = W_out @ s_t + b_out
    prediction = torch.argmax(logits)
    print(f"Sentence: {test_sentence} ➔ Prediction: {prediction.item()}")
