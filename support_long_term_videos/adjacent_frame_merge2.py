import torch
import torch.nn.functional as F
import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def merge_tokens(tensor):
    T, s, c = tensor.shape

    start_time1 = time.time()
    # Reshape tensor for batched cosine similarity calculation
    first_tokens = tensor[:-1]
    print(f"Shape of first_tokens:{first_tokens.shape}")
    second_tokens = tensor[1:]
    print(f"Shape of second_tokens:{second_tokens.shape}")
    end_time1 = time.time()

    start_time2 = time.time()
    # Calculate cosine similarity between all adjacent pairs
    similarities = F.cosine_similarity(first_tokens, second_tokens, dim=2).mean(dim=1)
    end_time2 = time.time()

    start_time3 = time.time()
    # Find the pair with the highest similarity
    merge_index = torch.argmax(similarities, dim=0)
    end_time3 = time.time()

    start_time4 = time.time()
    # Merge the two tokens with the highest similarity
    # new_token = (tensor[merge_index] + tensor[merge_index + 1]) / 2
    # tensor[merge_index].add_(tensor[merge_index + 1]).div_(2)

    # Gather the pair of tokens to merge
    tokens_to_merge = torch.index_select(tensor, 0, torch.tensor([merge_index, merge_index + 1], device=device))
    # Compute the mean of the pair
    new_token = tokens_to_merge.view(2, s, c).mean(dim=0)

    # Replace the token at merge_index with the mean
    tensor[merge_index] = new_token
    end_time4 = time.time()

    start_time5 = time.time()
    new_tensor = torch.cat((tensor[:merge_index + 1], tensor[merge_index + 2:]), dim=0)
    end_time5 = time.time()

    print("Time taken for 1: {:.2f} seconds".format(end_time1 - start_time1))
    print("Time taken for 2: {:.2f} seconds".format(end_time2 - start_time2))
    print("Time taken for 3: {:.2f} seconds".format(end_time3 - start_time3))
    print("Time taken for 4: {:.2f} seconds".format(end_time4 - start_time4))
    print("Time taken for 5: {:.2f} seconds".format(end_time5 - start_time5))

    return new_tensor


# Example tensor of dimensions [T, s, c]
T, s, c = 1000, 256, 1024  # Example dimensions, T should be more than 100
tensor = torch.rand(T, s, c).to(device)
print(f"Input tensor shape is:{tensor.shape}")

# Start the timer
start_time = time.time()

# Repeat merging until we have 100 tokens
T = tensor.size(0)
while T > 100:
    tensor = merge_tokens(tensor)
    T -= 1
    
# Stop the timer
end_time = time.time()

print(f"Output tensor shape is:{tensor.shape}")  # The final shape should be [100, s, c]

print("Time taken for the entire code: {:.2f} seconds".format(end_time - start_time))

