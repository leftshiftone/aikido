# aikido

Aikido provides state-of-the-art general-purpose ML models for various machine learning tasks.
It is based on pytorch and integrates further well known libraries.

**The framework is under development. There may be breaking changes in the design of the library in the upcoming commits.**

## Architecture
The framework uses the martial art of the same name for the naming. So aikido uses 
an analogy to describe the fundamental blocks of the library.

 * aikidoka: a predefined model which can be trained with data for a specific task
 * dojo: the component which trains an aikidoka
 * kun: the configuration (rules) of a dojo or an aikidoka
 * kata: the training data (routines) for the aikidoka

## Getting started

The first step to get aikido running is to define an aikidoka instance.

```python
from aikido.nn.modules.embedding import BPEmbEmbedding
from aikido.aikidoka.classification import AwdLstm, AwdLstmKun

embedding = BPEmbEmbedding("de")
aikidokaKun = AwdLstmKun(embedding, hidden_layers=4, hidden_size=128)
aikidoka = AwdLstm(aikidokaKun)
```

After that a dojo instance have to be instantiated.

``` python
optimizer = torch.optim.AdamW(aikidoka.parameters(), lr=1e-4)
loss = nn.CrossEntropyLoss()

dojoKun = RnnDojoKun(optimizer, loss, dans=20)
dojo = RnnDojo(dojoKun)
```

An aikidoka is trained with a dojo by a kata.

```python
from aikido.kata import CsvKata

kata = CsvKata.from_file('train.csv', embedding)
kata_train, kata_val = kata.split(0.8)
kata_test = CsvKata.from_file('test.csv', embedding)

dojo.train(aikidoka, kata_train)
```

After training the aikidoka can be evaluated.

````python
evaluation = dojo.evaluate(aikidoka, kata_val)

from aikido.visuals import AccuracyScore
AccuracyScore().render(evaluation)
````

## Event System

Aikido makes heavily use of decomposition. So much of the features can be
attached to the underlying event system to activate it.
For example in order to enable **learning rate optimization** a 
**LearningRateStepListener** has to be registered to the event system.

``` python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
dojo.add_listener(LearningRateStepListener(scheduler))
```
