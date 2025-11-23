

import os
import copy
import torch
import time
import torch.nn as nn
import datetime as dt
import pandas as pd
import numpy as np
import torchvision


from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from .model import CharRNN, device, cudnn, LSTMClassifier
from .utils import *
from .global_constants import *
from .reporter import Reporter

# https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
def rnn_languages_pytorch(
    ROOT: Path,
    n_epochs = 100000,
    n_hidden = 128,
    learning_rate = 0.005,
    plot_every = 1000,
    print_every = 5000,
    n_letters = len(NLPUtils.all_letters),
):
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS / f"rnn_languages_pytorch_{timestamp}"
    results_dir.mkdir(parents=True,exist_ok=True)
    
    # --------------- gathering the data first --------------- #
    # 1) accumalating all the data from all the files 
    # 2) split accross test and train for each category
    # 3) save for both test and train to a csv for later plotting and evaluation
    category_lines = {}
    all_categories = []
    datapath = ROOT / DATASETS / "languages/names"
    for filename in datapath.glob("*.txt"):
        category = filename.stem
        all_categories.append(category)
        lines = NLPUtils.readLines(filename)
        category_lines[category] = lines
    n_categories = len(all_categories)

    train_data = []
    test_data = []
    for category, lines in category_lines.items():
        train_lines, test_lines = train_test_split(
            lines, test_size=0.2, shuffle=True, random_state=RANDOM_SEED
        )

        for line in train_lines:
            train_data.append((category, line))

        for line in test_lines:
            test_data.append((category, line))

    train_df = pd.DataFrame(train_data, columns=["category", "line"])
    test_df = pd.DataFrame(test_data, columns=["category", "line"])

    Reporter.save__dataset_splits(
        train_data=train_df,
        test_data=test_df,
        results_dir=results_dir
    )
    Reporter.save__all_categories(
        all_categories=all_categories,
        results_dir=results_dir
    )
    # NOTE: we can see that the dataset is quite unbalanced (Russian has way more samples then others)
    # this will impact the performance of the model on the underrepresented classes
    # to mitigate this we choose to use stratified sampling during the train/test split
    Reporter.plot__datasets_distribution(
        train_data=train_df,
        test_data=test_df,
        results_dir=results_dir
    )
    Reporter.save__all_letters(
        all_letters=NLPUtils.all_letters,
        results_dir=results_dir
    )

    # --------------- init the model --------------- #
    model = CharRNN(
        input_size=n_letters, 
        hidden_size=n_hidden, 
        output_size=n_categories
    ).to(device)
    criterion = nn.NLLLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    if cudnn:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # --------------- start training loop --------------- #
    # 1) create input & target tensors
    # 2) reset gradients
    # 3) forward pass
    # 4) calculate loss
    # 5) backwards pass 
    # 6) optimize params
    start = time.time()
    all_losses = []
    current_loss = 0
    # creates a lookup table
    category_to_index = {cat: i for i, cat in enumerate(all_categories)}
    Reporter.save__to_index_map(
        to_index_map=category_to_index,
        results_dir=results_dir
    )

    for iter in range(1, n_epochs + 1):
        category, line, category_tensor, line_tensor = NLPUtils.randomSample(category_to_index,all_categories,category_lines)        

        optimizer.zero_grad()
        # forwards through the whole sequence at once instead 
        # of using the python for loop in the notebook which is faster
        output = model(line_tensor)         

        loss = criterion(output, category_tensor)
        loss.backward()
        current_loss += loss.item()
        optimizer.step()

        # --------------- reporting --------------- #
        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = NLPUtils.categoryFromOutput(output,all_categories)
            correct = '✓' if guess == category else f"✗ ({category})"
            print(
                f"epoch : {iter:<10} | ({iter / n_epochs:3.0%}) ({timeSince(start):>10})"
                f" | {loss:.2f} {line} {guess} {correct}"
            )
        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append((iter,current_loss / plot_every))
            current_loss = 0

    # --------------- save training results --------------- #
    Reporter.save__training_results(
        rnn=model,
        all_losses=all_losses,
        results_dir=results_dir,
        learning_rate=learning_rate,
        n_hidden=n_hidden,
        n_epochs=n_epochs,
        n_letters=n_letters,
        end=timeSince(start)
    )

    # --------------- plot losses --------------- #
    Reporter.plot__loss_over_time(
        results_dir=results_dir
    )

    # --------------- plot confusion matrix --------------- #
    conf = np.zeros((n_categories, n_categories), dtype=int)
    for _, row in test_df.iterrows():
        category = row["category"]
        line_tensor = NLPUtils.lineToTensor(row["line"])
        # forward pass to get a prediction
        with torch.no_grad():
            output = model(line_tensor)

        guess, guess_i = NLPUtils.categoryFromOutput(output, all_categories)
        true_i = category_to_index[category]
        conf[true_i][guess_i] += 1

    confusion_matrix_df = pd.DataFrame(
        conf,
        index=all_categories,
        columns=all_categories
    )
    Reporter.save_confusion_matrix(
        confusion_matrix=confusion_matrix_df,
        results_dir=results_dir
    )
    Reporter.plot__confusion_matrix(
        results_dir=results_dir
    )

# ////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////

def custom_languages_lstm_pytorch(
    ROOT: Path,
    n_epochs=5,
    lr=0.001,
    hidden_size=256,
    batch_size=64,
    max_len=256
):
    # using : https://www.kaggle.com/datasets/tanishqdublish/text-classification-documentation
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS / f"lstm_document_classifier_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------- gathering the data first --------------- #
    df = pd.read_csv(ROOT / DATASETS / "documentation_classification" / "Documentation.csv")
    df = df.rename(columns={"Text": "line", "Label": "category"})
    df = df.dropna()
    # Politics = 0
    # Sport = 1
    # Technology = 2
    # Entertainment =3
    # Business = 4

    all_categories = sorted(df["category"].unique())
    category_to_index = {int(c): i for i, c in enumerate(all_categories)}
    # map key number to string label
    index_to_category = {i: int(c) for i, c in enumerate(all_categories)}

    n_categories = len(all_categories)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, shuffle=True, stratify=df["category"]
    )

    Reporter.save__dataset_splits(train_df, test_df, results_dir)
    Reporter.save__all_categories(all_categories, results_dir)
    Reporter.save__to_index_map(category_to_index, results_dir)
    Reporter.plot__datasets_distribution(train_df, test_df, results_dir)

    tokenizer = SpacyTokenizer()
    token_lists = [tokenizer(text) for text in train_df["line"].tolist()]

    vocab = Vocabulary(min_freq=2)
    vocab.build(token_lists)
    vocab_size = len(vocab.idx2word)
    Reporter.save__vocabulary(
        vocab=vocab,
        results_dir=results_dir
    )
    train_ds = TextDataset(
        texts=train_df["line"].tolist(),
        labels=[category_to_index[c] for c in train_df["category"].tolist()],
        tokenizer=tokenizer,
        vocab=vocab
    )
    test_ds = TextDataset(
        texts=test_df["line"].tolist(),
        labels=[category_to_index[c] for c in test_df["category"].tolist()],
        tokenizer=tokenizer,
        vocab=vocab
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False
    )

    # --------------- init the model --------------- #
    model = LSTMClassifier(
        vocab_size=vocab_size, 
        embed_dim=128,
        output_dim=n_categories,
        hidden_dim=hidden_size,
        vocab=vocab
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if cudnn:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # --------------- start training loop --------------- #
    start = time.time()
    all_losses = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        all_losses.append((epoch, avg_loss))

        print(f"Epoch {epoch}/{n_epochs} | Loss {avg_loss:.4f}")

    end = time.time()

    Reporter.save__training_results(
        rnn=model,
        all_losses=all_losses,
        results_dir=results_dir,
        learning_rate=lr,
        n_hidden=hidden_size,
        n_epochs=n_epochs,
        n_letters=vocab_size,
        end=timeSince(start),
    )

    Reporter.plot__loss_over_time(results_dir)

    conf = np.zeros((n_categories, n_categories), dtype=int)

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for t, p in zip(y.numpy(), preds):
                conf[t][p] += 1

    df_cm = pd.DataFrame(conf, index=all_categories, columns=all_categories)

    Reporter.save_confusion_matrix(df_cm, results_dir)
    Reporter.plot__confusion_matrix(results_dir)


# ////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////
from tensorboard import program

def launch_tensorboard(logdir, port=6006):
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", str(logdir), "--port", str(port)])
    url = tb.launch()
    return url

def cnn_fashion_mnist_pytorch(
    ROOT: Path,
    n_epochs = 1,
    learning_rate = 0.001,
    momentum = 0.9,
    log_interval = 2000,
):
    # NOTE: Tnesorboard has some usefull visualizations but i am not really for it
    #       prefer using matplotlib/seaborn for reporting and visualizations
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.tensorboard import SummaryWriter
    from .model import Net, device, cudnn
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS / f"cnn_fashion_mnist_pytorch_{timestamp}"
    results_dir.mkdir(parents=True,exist_ok=True)
  
    # --------------- gathering the data first --------------- #
    # Gather datasets and prepare them for consumption
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))]
    )

    # Store separate training and validations splits in ./data
    training_set = torchvision.datasets.FashionMNIST(
        root=DATASETS,
        download=True,
        train=True,
        transform=transform
    )
    validation_set = torchvision.datasets.FashionMNIST(
        root=DATASETS,
        download=True,
        train=False,
        transform=transform
    )

    training_loader = torch.utils.data.DataLoader(
        training_set,  # It is an iterable that will be used as a batch loader and load 4 images.
        batch_size=4,
        shuffle=True,
        num_workers=1
    )  # For windows num_workers should be set to 1.

    validation_loader = torch.utils.data.DataLoader(
        validation_set, # It is an iterable that will be used as a batch loader and load 4 images.
        batch_size=4,
        shuffle=False,
        num_workers=1
    ) # For windows num_workers should be set to 1.

    # Class labels
    classes = (
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle Boot'
    )

    # --------------- init the model --------------- #
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    if cudnn:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # --------------- log model graph and some images --------------- #

    log_dir = LOGS_DIR / f'fashion_mnist_experiment_1_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)

    # only for testing purposes
    # url = launch_tensorboard(logdir=ROOT / LOGS_DIR, port=6006)
    # print("TensorBoard running at:", url)
    
    dataiter = iter(training_loader)
    images, labels = next(dataiter)
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image(
        tag='four_fashion_mnist_images', 
        img_tensor=img_grid
    )

    writer.add_graph(
        model=model,
        input_to_model=images.to(device)
    )

    # --------------- start training loop --------------- #
    # loop over the dataset multiple times
    training_losses = []
    validation_losses = []
    start = time.time()
    for epoch in range(n_epochs):  
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(training_loader, 0):
            # 28x28 grayscale images => 1x28x28 CHW
            # and labels are 0-9
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # forward + calculate loss + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # --------------- reporting --------------- #
            # log every log_interval mini-batches to tensorboard and terminal
            if i % log_interval == (log_interval - 1):    
                
                remaining = (len(training_loader) - (i + 1))
                pct = (i + 1) / len(training_loader)
                avg_loss = running_loss / 2000
                training_losses.append((epoch * len(training_loader) + i, avg_loss))
                running_loss = 0.0
                
                # validation loss
                running_vloss = 0.0
                model.eval()
                for (vinputs, vlabels) in validation_loader:
                    vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                    voutputs = model(vinputs)
                    vloss = criterion(voutputs, vlabels)
                    running_vloss += vloss.item()
                model.train()
                avg_vloss = running_vloss / len(validation_loader)
                validation_losses.append((epoch * len(training_loader) + i, avg_vloss))
                
                print(
                    f"[Epoch {epoch+1:<2} | {i +1:5d} | ({timeSince(start):>7})] | remaining: {remaining} ({pct:5.2%})  | AVG loss: {avg_loss:.3f}"
                )
                writer.add_scalar(
                    tag='training_loss',
                    scalar_value=avg_loss,
                    global_step=epoch * len(training_loader) + i
                )
                
                print(
                    f"[Epoch {epoch+1:<2}, {i +1:5d}] | AVG validation loss: {avg_vloss:.3f}"
                )
                writer.add_scalars(
                    main_tag='Training_vs_Validation_Loss',
                    tag_scalar_dict={
                        'validation_loss': avg_vloss,
                        'training_loss': avg_loss
                    },
                    global_step=epoch * len(training_loader) + i
                )
                
        print(f"Epoch {epoch+1} completed in {timeSince(start)}")
    end = time.time()


    Reporter.save__training_results_cnn(
        model=model,
        results_dir=results_dir,
        n_epochs=n_epochs,
        end=timeSince(start),
    )

    # --------------- save training results --------------- #
    def select_n_random(data, labels, n=100):
        assert len(data) == len(labels)

        perm = torch.randperm(len(data))
        return data[perm][:n], labels[perm][:n]

    images, labels = select_n_random(training_set.data, training_set.targets)
    class_labels = [classes[label] for label in labels]

    features = images.view(-1, 28 * 28)
    writer.add_embedding(
        features,
        metadata=class_labels,
        label_img=images.unsqueeze(1)
    )    

    writer.add_hparams(
        hparam_dict={
            'n_epochs': n_epochs,
            'learning_rate': 0.001,
            'optimizer': 'SGD',
            'loss': 'CrossEntropyLoss',
        },
        metric_dict={
            'final_validation_loss': avg_vloss,
            'total_training_time': end - start,
        }
    )

    print('Finished Training')
    writer.flush()
    writer.close()

    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for (vinputs, vlabels) in validation_loader:
            vinputs = vinputs.to(device)
            outputs = model(vinputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(vlabels.numpy())

    cm = confusion_matrix(all_targets, all_preds)

    Reporter.save__cnn_fashion_mnist_results(
        results_dir=results_dir,
        training_losses_df=pd.DataFrame(training_losses, columns=["step","losses"]),
        validation_losses_df=pd.DataFrame(validation_losses, columns=["step","losses"]),
        confusion_matrix=pd.DataFrame(cm, index=classes, columns=classes)
    )
    Reporter.plot__cnn_fashion_mnist_results(
        results_dir=results_dir,
        confusion_matrix=pd.DataFrame(cm, index=classes, columns=classes)
    )

# ////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////

def cnn_beexants_pytorch(
    ROOT: Path,
    n_epochs = 10,
    learning_rate = 0.001,
    momentum = 0.9,
):
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS / f"cnn_beexants_pytorch_{timestamp}"
    results_dir.mkdir(parents=True,exist_ok=True)

    # --------------- gathering the data first --------------- #

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    image_datasets = {
        x : datasets.ImageFolder(
            root=ROOT / DATASETS / "data_beexant" / x,
            transform=data_transforms[x]
        ) for x in ['train', 'val']
    }
    dataloaders = {
        x : torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=4,
            shuffle=True,
            num_workers=4
        ) for x in ['train', 'val']
    }

    dataset_sizes = {
        x: len(image_datasets[x]) for x in ['train', 'val']
    }
    class_names = image_datasets['train'].classes

    # --------------- init the model --------------- #
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    # freeze all the layers and reset the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.1
    )

    # --------------- start training loop --------------- #
    model_best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    losses = {'train': [], 'val': []}
    accuracy = {'train': [], 'val': []}

    start = time.time()
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}/{n_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            losses[phase].append((epoch, epoch_loss))
            accuracy[phase].append((epoch, epoch_acc.item()))

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                model_best_wts = copy.deepcopy(model.state_dict())

    end = time.time()

    confusion_matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1


    Reporter.save__cnn_beexants_results(
        best_model_wts=model_best_wts,
        results_dir=results_dir,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        end=timeSince(start),
        training_losses_df=pd.DataFrame(losses['train'], columns=["epoch","losses"]),
        validation_losses_df=pd.DataFrame(losses['val'], columns=["epoch","losses"]),
        confusion_matrix=pd.DataFrame(
            confusion_matrix,
            index=class_names,
            columns=class_names
        ),
        accuracy_training_df=pd.DataFrame(accuracy['train'], columns=["epoch","accuracy"]),
        accuracy_validation_df=pd.DataFrame(accuracy['val'], columns=["epoch","accuracy"])
    )
    Reporter.plot__cnn_beexants_results(
        results_dir=results_dir,
        confusion_matrix=pd.DataFrame(
            confusion_matrix,
            index=class_names,
            columns=class_names
        )
    )
