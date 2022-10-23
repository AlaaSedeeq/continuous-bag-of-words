data_prep = DataPrep('Data/t8.shakespeare.txt',243, 300)
data_prep.build_dictionary()
vocab_size = data_prep.vocab_size
corpus = data_prep.corpus

# define the mdoel
model = model = CBOW(
    embedding_dim=100,
    input_dim=vocab_size,
    window_size=5,
    enc_dim=[1024, 1024],
    ls_dim=[512, 512],
    dec_dim=[1024, 1024],
    enc_dropout=0.5,
    dec_dropout=0.5,
    ls_dropout=0.5
)
    
# define training criterion
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr)

for epoch in range(epochs):
    prog_bar = tqdm(range(len(corpus) - window_size - 1))
    for i in prog_bar:
        # apply sliding window, where the length is odd number
        # y is the middle word, x is the right and left words
        X = data_prep.Decode(corpus[i:i+window_size//2] + corpus[i+window_size//2+1:i+window_size]).unsqueeze(0).float()
        y = data_prep.Decode(corpus[i+window_size//2])
        
        ################
        # Forward Pass #
        ################
        out = model(X)
        # compute loss
        loss = criterion(out.view_as(y), y.float())

        #################
        # Backward Pass #
        #################
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check point
        prog_bar.set_description('Epoch {}/{}, Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

    torch.save(model.state_dict(), model_path+'_epoch_{}'.format(epoch+1))
