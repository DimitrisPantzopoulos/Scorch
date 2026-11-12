from tensorboardX import SummaryWriter

training_run : str = "test_run"
logging_path : str = 'runs/logging_test'

writer : SummaryWriter = SummaryWriter(log_dir=logging_path)

for i in range(5):
    writer.add_scalar('test/value', i, i)

writer.close()