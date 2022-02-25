def evaluate(eval_dataset, model, device="cpu"):
    model.load_state_dict(torch.load("parameters.pt"))
    model.eval()

    eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False)

    count = 0
    for i_batch, batch in enumerate(eval_dataloader):
        n_batch = batch.batch.max() + 1
        # now, batch contains a mega graph containing each
        # graph in the batch, as usual with pyg data loaders.
        # each of these graphs has a 'descs' array containing
        # its descriptions, all of which get combined into one
        # giant nested array. we tokenize them below:
        batch.to(device)
        # we could honestly move this code into the model's forward
        # function now that we're using pyg
        batch_texts = torch.cat([model.tokenizer(model_descs, return_tensors="pt", padding='max_length',
                                                 truncation=True).input_ids
                                 for model_descs in batch.descs], dim=0).to(device)
        # vector mapping each description to its mesh index
        desc2mesh = torch.zeros(batch_texts.shape[0], dtype=torch.long)
        # uniform distribution over matching descs
        target_per_mesh = torch.zeros(n_batch, batch_texts.shape[0]).to(device)
        # one-hot distribution for single matching shape
        target_per_text = torch.zeros(batch_texts.shape[0], n_batch).to(device)
        # loop over the descriptions and populate above
        i_desc = 0
        for i_mesh, model_descs in enumerate(batch.descs):
            desc2mesh[i_desc:i_desc + len(model_descs)] = i_mesh
            target_per_text[i_desc:i_desc + len(model_descs), i_mesh] = 1
            i_desc += len(model_descs)

        logits_per_mesh, logits_per_text = model(batch, batch_texts, desc2mesh)
        eval_acc = eval(logits_per_text, target_per_text)
        writer.add_scalar('Accu/eval', acc.item(), count)
        print('eval accuracy:', eval_acc)
        count += 1