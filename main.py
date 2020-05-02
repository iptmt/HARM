import sys, json, torch
from data_util import TrainLoader, TestLoader
from util import LossClock, mrr, create_logger, max_margin_loss
from model import HARM_Model


# Hyper-parameters
total_epochs = 10
lr = 2.0
device = "CUDA:0"
vocab_file = "dump/vocab.json"
glove_emb_file = "dump/glove.emb"
model_dump_file = "dump/model.pth"

train_file = "data/train.csv"
dev_file = "data/train.csv"
test_file = "data/train.csv"


mode = sys.argv[1]
if mode != "train" and mode != "test":
    raise ValueError

# Running
with open(vocab_file, 'r') as f:
    vocab = json.load(f)
glove_emb = torch.load(glove_emb_file)
model = HARM_Model(len(vocab), glove_emb).to(device)

if mode == "train":
    best_mrr = 0.
    logger = create_logger("log/", "train.log")

    optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    clock = LossClock(["loss"], interval=20)

    ds_train = TrainLoader(train_file, vocab, device)
    ds_dev = TestLoader(dev_file, vocab, device)

    for epoch in range(total_epochs):
        # train
        logger.info("=" * 50 + f"Train epoch {epoch}" + "=" * 50)
        for query, docs in ds_train():
            r = model(query, docs)
            margin_loss = max_margin_loss(r[:1].expand(r[1:].size(0)), r[1:])
            # update
            optimizer.zero_grad()
            margin_loss.backward()
            optimizer.step()
            clock.update({"loss": margin_loss.item()})
        # evaluate
        logger.info("=" * 50 + f"Evaluate epoch {epoch}" + "=" * 50)
        rs, ls = [], []
        with torch.no_grad():
            for query, docs, label in ds_dev():
                r = model(query, docs)
                rs.append(r)
                ls.append(label)
        mrr_score = mrr(rs, ls)
        if mrr_score > best_mrr:
            logger.info(f"Best result {best_mrr} -> {mrr_score}.")
            torch.save(model.state_dict(), model_dump_file)
            best_mrr = mrr_score
        else:
            logger.info(f"Skip the result {mrr_score} < {best_mrr}.")
            

if mode == "test":
    ds_test = TestLoader(test_file, vocab, device)
    model.load_state_dict(torch.load(model_dump_file))

    ds_test = TestLoader(test_file, vocab, device)

    rs, ls = [], []
    with torch.no_grad():
        for query, docs, label in ds_test():
            r = model(query, docs)
            rs.append(r)
            ls.append(label)
    #TODO: write the predictions to output file
    print("Done.")