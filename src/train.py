import logging
from config import config
import torch
from model_io import dump_model

logger = logging.getLogger(__name__)


# encoding function for fine-tuning on paws
def encode_sentence_pair_batch(tokenizer, sequence_batch):
    tuple_batch = []
    batch_size = len(sequence_batch[0])
    
    assert batch_size == len(sequence_batch[1])  # make sure number of sentence 1 is the same as sentence 2

    # model_max_len = tokenizer.max_model_input_sizes[pretrained_name]
    # process sentence batch from loader
    for ind in range(batch_size):
        tuple_batch.append((sequence_batch[0][ind], sequence_batch[1][ind]))

    # encoded_info = tokenizer.batch_encode_plus(tuple_batch, \
    #                             return_tensors='pt', pad_to_max_length=True, max_length=model_max_len)
    max_len = min(512, tokenizer.model_max_length)
    encoded_info = tokenizer(tuple_batch, \
                                return_tensors='pt', padding=True, truncation=True, max_length=max_len)
    input_tensor, attention_tensor = encoded_info['input_ids'], encoded_info['attention_mask']

    return input_tensor, attention_tensor


# encoding function for fine-tuning on stanford sentiment treebank
def encode_sentence_batch(tokenizer, sequence_batch):
    max_len = min(512, tokenizer.model_max_length)
    encoded_info = tokenizer(sequence_batch, \
                                return_tensors='pt', padding=True, truncation=True, max_length=max_len)
    input_tensor, attention_tensor = encoded_info['input_ids'], encoded_info['attention_mask']
    return input_tensor, attention_tensor


# wrapper to call the correct encoding function
def encode_batch(tokenizer, sequence_batch, task):
    if task == "paws":
        return encode_sentence_pair_batch(tokenizer, sequence_batch)
    elif task == "standsent":
        return encode_sentence_batch(tokenizer, sequence_batch)
    else:
        logger.error("unsupported task: {}".format(task))
        exit(1)


def evaluate(model, tokenizer, data_loader, task, error_analysis=False):
    logger.info("evaluating...")
    model.eval()
    errors = []
    correct_predictions = []
    with torch.no_grad():
        output_str = "workload size {}, precision {}"
        hit, total = 0.0, 0.0

        for sentence_batch, label_batch in data_loader:
            input_tensor, attention_tensor = encode_batch(tokenizer, sentence_batch, task)
            output_tensor = model(input_tensor, attention_mask=attention_tensor)
            logits = output_tensor[0]
            prediction = torch.topk(logits, 1).indices.reshape(-1)  # prediction reshape from (batch_size, 1) to (batch_size) to match label_batch shape
            assert prediction.shape == label_batch.shape

            results = (prediction == label_batch)
            cur_hit = sum(results).item()
            hit += cur_hit
            total += label_batch.size(0)

            if error_analysis:
                for index, result in enumerate(results):
                    current_sample = (sentence_batch[0][index], sentence_batch[1][index], label_batch[index].item())
                    if result.item() is False:
                        # incorrect prediction
                        errors.append(current_sample)
                    else:
                        # correct prediction
                        correct_predictions.append(current_sample)


        precision = hit / total
        logger.info(output_str.format(total, precision))

    model.train()    
    return precision, (errors, correct_predictions)
    


def train(model, input_tensor, attention_tensor, label_tensor, optimizer):
    optimizer.zero_grad()
    out = model(input_tensor, attention_mask=attention_tensor, labels=label_tensor)
    loss, logits = out[:2]
    loss.backward()
    optimizer.step()
    return loss.item()


def train_iter(model, tokenizer, optimizer, train_loader, dev_loader, task, early_stopping, max_epochs=1, print_every=50, evaluate_every=500, init_eval=True):
    CONVERGENCE_THRESH = 0.01
    total_loss = 0.0
    output_str = "epoch {}, iteration {}, loss {}"
    best_prec, prev_prec, init_loss, best_loss = None, None, None, None
    best_errors = None # record incorrect predictions made by the best model by far
    best_model = None
    max_iter = -1 # for debug purpose
    optimizer.zero_grad()
    breaking_flag = False

    if init_eval:
        init_prec, best_errors = evaluate(model, tokenizer, dev_loader, task)
    else:
        init_prec = 0.0
    
    for epoch in range(1, max_epochs + 1):
        for iter, (sentence_batch, label_batch) in enumerate(train_loader, 1):
            if iter == max_iter:
                break

            input_tensor, attention_tensor = encode_batch(tokenizer, sentence_batch, task)
            loss = train(model, input_tensor, attention_tensor, label_batch, optimizer)
            logger.debug("loss = {}".format(loss))
            total_loss += loss
            
            if iter % print_every == 0:
                avg_loss = total_loss / print_every
                logger.info(output_str.format(epoch, iter, avg_loss))
                total_loss = 0.0
                if init_loss is None:
                    init_loss = avg_loss
                    best_loss = avg_loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                

            if iter % evaluate_every == 0:
                precision, cur_errors = evaluate(model, tokenizer, dev_loader, task)
                if (best_prec is None) or (precision > best_prec):
                    logger.info("dumping model with precision = {}...".format(precision))
                    dump_model(model)
                    best_model = model
                    best_prec = precision
                    best_errors = cur_errors
                    logger.info("dumping done.")
                # test early stopping condition
                if early_stopping and (epoch > 1) and (precision - prev_prec <= CONVERGENCE_THRESH):
                    logger.info("early stopping...")
                    breaking_flag = True
                    break
                prev_prec = precision
        if breaking_flag:
            break

    # post training evaluation
    # since workload size mode evluate_every might not be integer
    precision, cur_errors = evaluate(model, tokenizer, dev_loader, task)
    if (best_prec is None) or (precision > best_prec):
        logger.info("dumping model with precision = {}...".format(precision))
        dump_model(model)
        best_model = model
        best_prec = precision
        best_errors = cur_errors
        logger.info("dumping done.")
    
    return init_prec, init_loss, best_prec, best_loss, best_model