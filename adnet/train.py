import time
from data.data_loader import CreateDataLoader
from adnet import ADNET
from util.visualizer import Visualizer

data_loader = CreateDataLoader()
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print(dataset_size)

model = adnet.ADNET()
total_steps = 0

for epoch in range(0, 100000):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += 100
        epoch_iter += 100
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % 10 == 0:
            print('save (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % 100 == 0:
        print('save (epoch %d, iters %d )' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d Time Taken: %d sec' %
          (epoch, time.time() - epoch_start_time))
    model.update_learning_rate()
