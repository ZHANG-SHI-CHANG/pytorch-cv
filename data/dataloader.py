# TODO: NOTE this is for pytorch1.0
# import sys
# import random
# import queue
# import threading
# import multiprocessing
#
# import torch
# from torch.utils.data.dataloader import DataLoader, ManagerWatchdog, MP_STATUS_CHECK_INTERVAL, ExceptionWrapper, \
#     _pin_memory_loop, _set_SIGCHLD_handler, pin_memory_batch, default_collate, _python_exit_status
#
# from torch._C import _set_worker_signal_handlers, _update_worker_pids, \
#     _remove_worker_pids
#
#
# # TODO need rewrite
# def _worker_loop(index_queue, data_queue, done_event,
#                  seed, init_fn, worker_id, cnt):
#     # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
#     # logic of this function.
#
#     try:
#         global _use_shared_memory
#         _use_shared_memory = True
#
#         # Intialize C side signal handlers for SIGBUS and SIGSEGV. Python signal
#         # module's handlers are executed after Python returns from C low-level
#         # handlers, likely when the same fatal signal happened again already.
#         # https://docs.python.org/3/library/signal.html Sec. 18.8.1.1
#         _set_worker_signal_handlers()
#
#         torch.set_num_threads(1)
#         random.seed(seed)
#         torch.manual_seed(seed)
#
#         data_queue.cancel_join_thread()
#
#         if init_fn is not None:
#             init_fn(worker_id)
#
#         watchdog = ManagerWatchdog()
#
#         while watchdog.is_alive():
#             try:
#                 r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
#             except queue.Empty:
#                 continue
#             if r is None:
#                 # Received the final signal
#                 assert done_event.is_set()
#                 return
#             elif done_event.is_set():
#                 # Done event is set. But I haven't received the final signal
#                 # (None) yet. I will keep continuing until get it, and skip the
#                 # processing steps.
#                 continue
#             idx, batch_indices = r
#             try:
#                 samples = cnt.increment(batch_indices)
#                 # if cnt.val.value % interval == 0:
#                 #     print('change')
#                 #     dataset.transform(np.random.choice(transform_fns))
#                 # samples = collate_fn([dataset[i] for i in batch_indices])
#                 # print(cnt.val.value)
#             except Exception:
#                 # It is important that we don't store exc_info in a variable,
#                 # see NOTE [ Python Traceback Reference Cycle Problem ]
#                 data_queue.put((idx, ExceptionWrapper(sys.exc_info())))
#             else:
#                 data_queue.put((idx, samples))
#                 del samples
#     except KeyboardInterrupt:
#         # Main process will raise KeyboardInterrupt anyways.
#         pass
#
#
# class Counter(object):
#     def __init__(self, dataset, transform_fns, collate_fn, interval, initval=0):
#         self.val = multiprocessing.Value('i', initval)
#         self.interval = interval
#         self.dataset = dataset
#         self.transform_fns = transform_fns
#         self.collate_fn = collate_fn
#         self.num = len(self.transform_fns)
#         self.lock = multiprocessing.Lock()
#
#     def increment(self, batch_indices):
#         with self.lock:
#             self.val.value += 1
#             if self.val.value % self.interval == 0:
#                 self.dataset.transform(self.transform_fns[random.randint(0, self.num - 1)])
#             return self.collate_fn([self.dataset[i] for i in batch_indices])
#
#     def value(self):
#         with self.lock:
#             return self.val.value
#
#
# class _MultiDataLoaderIter(object):
#     def __init__(self, loader, transform_fns, interval):
#         self.dataset = loader.dataset
#         self.collate_fn = loader.collate_fn
#         self.batch_sampler = loader.batch_sampler
#         self.num_workers = loader.num_workers
#         self.pin_memory = loader.pin_memory and torch.cuda.is_available()
#         self.timeout = loader.timeout
#
#         self.sample_iter = iter(self.batch_sampler)
#
#         base_seed = torch.LongTensor(1).random_().item()
#
#         if self.num_workers > 0:
#             self.worker_init_fn = loader.worker_init_fn
#             self.worker_queue_idx = 0
#             self.worker_result_queue = multiprocessing.Queue()
#             self.batches_outstanding = 0
#             self.worker_pids_set = False
#             self.shutdown = False
#             self.send_idx = 0
#             self.rcvd_idx = 0
#             self.reorder_dict = {}
#             self.done_event = multiprocessing.Event()
#
#             self.index_queues = []
#             self.workers = []
#             cnt = Counter(self.dataset, transform_fns, self.collate_fn, interval)
#             for i in range(self.num_workers):
#                 index_queue = multiprocessing.Queue()
#                 index_queue.cancel_join_thread()
#                 w = multiprocessing.Process(
#                     target=_worker_loop,
#                     args=(index_queue, self.worker_result_queue,
#                           self.done_event, base_seed + i,
#                           self.worker_init_fn, i, cnt))
#                 w.daemon = True
#                 # NB: Process.start() actually take some time as it needs to
#                 #     start a process and pass the arguments over via a pipe.
#                 #     Therefore, we only add a worker to self.workers list after
#                 #     it started, so that we do not call .join() if program dies
#                 #     before it starts, and __del__ tries to join but will get:
#                 #     AssertionError: can only join a started process.
#                 w.start()
#                 self.index_queues.append(index_queue)
#                 self.workers.append(w)
#
#             if self.pin_memory:
#                 self.data_queue = queue.Queue()
#                 pin_memory_thread = threading.Thread(
#                     target=_pin_memory_loop,
#                     args=(self.worker_result_queue, self.data_queue,
#                           torch.cuda.current_device(), self.done_event))
#                 pin_memory_thread.daemon = True
#                 pin_memory_thread.start()
#                 # Similar to workers (see comment above), we only register
#                 # pin_memory_thread once it is started.
#                 self.pin_memory_thread = pin_memory_thread
#             else:
#                 self.data_queue = self.worker_result_queue
#
#             _update_worker_pids(id(self), tuple(w.pid for w in self.workers))
#             _set_SIGCHLD_handler()
#             self.worker_pids_set = True
#
#             # prime the prefetch loop
#             for _ in range(2 * self.num_workers):
#                 self._put_indices()
#
#     def __len__(self):
#         return len(self.batch_sampler)
#
#     def _get_batch(self):
#         # In the non-timeout case, worker exit is covered by SIGCHLD handler.
#         # But if `pin_memory=True`, we still need account for the possibility
#         # that `pin_memory_thread` dies.
#         if self.timeout > 0:
#             try:
#                 return self.data_queue.get(timeout=self.timeout)
#             except queue.Empty:
#                 raise RuntimeError('DataLoader timed out after {} seconds'.format(self.timeout))
#         elif self.pin_memory:
#             while self.pin_memory_thread.is_alive():
#                 try:
#                     return self.data_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
#                 except queue.Empty:
#                     continue
#             else:
#                 # while condition is false, i.e., pin_memory_thread died.
#                 raise RuntimeError('Pin memory thread exited unexpectedly')
#             # In this case, `self.data_queue` is a `queue.Queue`,. But we don't
#             # need to call `.task_done()` because we don't use `.join()`.
#         else:
#             return self.data_queue.get()
#
#     def __next__(self):
#         if self.num_workers == 0:  # same-process loading
#             indices = next(self.sample_iter)  # may raise StopIteration
#             batch = self.collate_fn([self.dataset[i] for i in indices])
#             if self.pin_memory:
#                 batch = pin_memory_batch(batch)
#             return batch
#
#         # check if the next sample has already been generated
#         if self.rcvd_idx in self.reorder_dict:
#             batch = self.reorder_dict.pop(self.rcvd_idx)
#             return self._process_next_batch(batch)
#
#         if self.batches_outstanding == 0:
#             self._shutdown_workers()
#             raise StopIteration
#
#         while True:
#             assert (not self.shutdown and self.batches_outstanding > 0)
#             idx, batch = self._get_batch()
#             self.batches_outstanding -= 1
#             if idx != self.rcvd_idx:
#                 # store out-of-order samples
#                 self.reorder_dict[idx] = batch
#                 continue
#             return self._process_next_batch(batch)
#
#     next = __next__  # Python 2 compatibility
#
#     def __iter__(self):
#         return self
#
#     def _put_indices(self):
#         assert self.batches_outstanding < 2 * self.num_workers
#         indices = next(self.sample_iter, None)
#         if indices is None:
#             return
#         self.index_queues[self.worker_queue_idx].put((self.send_idx, indices))
#         self.worker_queue_idx = (self.worker_queue_idx + 1) % self.num_workers
#         self.batches_outstanding += 1
#         self.send_idx += 1
#
#     def _process_next_batch(self, batch):
#         self.rcvd_idx += 1
#         self._put_indices()
#         if isinstance(batch, ExceptionWrapper):
#             raise batch.exc_type(batch.exc_msg)
#         return batch
#
#     def __getstate__(self):
#         # TODO: add limited pickling support for sharing an iterator
#         # across multiple threads for HOGWILD.
#         # Probably the best way to do this is by moving the sample pushing
#         # to a separate thread and then just sharing the data queue
#         # but signalling the end is tricky without a non-blocking API
#         raise NotImplementedError("_DataLoaderIter cannot be pickled")
#
#     def _shutdown_workers(self):
#         # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on
#         # the logic of this function.
#         if _python_exit_status is True or _python_exit_status is None:
#             # See (2) of the note. If Python is shutting down, do no-op.
#             return
#         # Normal exit when last reference is gone / iterator is depleted.
#         # See (1) and the second half of the note.
#         if not self.shutdown:
#             self.shutdown = True
#             # Removes pids from the C side data structure first so worker
#             # termination afterwards won't trigger false positive error report.
#             if self.worker_pids_set:
#                 _remove_worker_pids(id(self))
#                 self.worker_pids_set = False
#
#             self.done_event.set()
#
#             # Exit `pin_memory_thread` first because exiting workers may leave
#             # corrupted data in `worker_result_queue` which `pin_memory_thread`
#             # reads from.
#             if hasattr(self, 'pin_memory_thread'):
#                 # Use hasattr in case error happens before we set the attribute.
#                 # First time do `worker_result_queue.put` in this process.
#
#                 # `cancel_join_thread` in case that `pin_memory_thread` exited.
#                 self.worker_result_queue.cancel_join_thread()
#                 self.worker_result_queue.put(None)
#                 self.pin_memory_thread.join()
#                 # Indicate that no more data will be put on this queue by the
#                 # current process. This **must** be called after
#                 # `pin_memory_thread` is joined because that thread shares the
#                 # same pipe handles with this loader thread. If the handle is
#                 # closed, Py3 will error in this case, but Py2 will just time
#                 # out even if there is data in the queue.
#                 self.worker_result_queue.close()
#
#             # Exit workers now.
#             for q in self.index_queues:
#                 q.put(None)
#                 # Indicate that no more data will be put on this queue by the
#                 # current process.
#                 q.close()
#             for w in self.workers:
#                 w.join()
#
#     def __del__(self):
#         if self.num_workers > 0:
#             self._shutdown_workers()
#
#
# class RandomTransformDataLoader(DataLoader):
#     """DataLoader that support random transform function applied to dataset.
#
#     Parameters
#     ----------
#     transform_fns : iterable of callables
#         Transform functions that takes a sample as input and returns the transformed sample.
#         They will be randomly selected during the dataloader iteration.
#     dataset : mxnet.gluon.data.Dataset or numpy.ndarray or mxnet.ndarray.NDArray
#         The source dataset. Original dataset is recommended here since we will apply transform
#         function from candidates again during the iteration.
#     interval : int, default is 1
#         For every `interval` batches, transform function is randomly selected from candidates.
#     batch_size : int
#         The size of mini-batch.
#     shuffle : bool, default False
#         If or not randomly shuffle the samples. Often use True for training
#         dataset and False for validation/test datasets
#     sampler : mxnet.gluon.data.Sampler, default None
#         The sampler to use. We should either specify a sampler or enable
#         shuffle, not both, because random shuffling is a sampling method.
#     last_batch : {'keep', 'discard', 'rollover'}, default is keep
#         How to handle the last batch if the batch size does not evenly divide by
#         the number of examples in the dataset. There are three options to deal
#         with the last batch if its size is smaller than the specified batch
#         size.
#
#         - keep: keep it
#         - discard: throw it away
#         - rollover: insert the examples to the beginning of the next batch
#     batch_sampler : mxnet.gluon.data.BatchSampler
#         A sampler that returns mini-batches. Do not specify batch_size,
#         shuffle, sampler, and last_batch if batch_sampler is specified.
#     batchify_fn : callable
#         Callback function to allow users to specify how to merge samples
#         into a batch.
#         Defaults to :py:meth:`gluoncv.data.dataloader.default_pad_batchify_fn`::
#             def default_pad_batchify_fn(data):
#                 if isinstance(data[0], nd.NDArray):
#                     return nd.stack(*data)
#                 elif isinstance(data[0], tuple):
#                     data = zip(*data)
#                     return [pad_batchify(i) for i in data]
#                 else:
#                     data = np.asarray(data)
#                     pad = max([l.shape[0] for l in data])
#                     buf = np.full((len(data), pad, data[0].shape[-1]),
#                                   -1, dtype=data[0].dtype)
#                     for i, l in enumerate(data):
#                         buf[i][:l.shape[0], :] = l
#                     return nd.array(buf, dtype=data[0].dtype)
#     num_workers : int, default 0
#         The number of multiprocessing workers to use for data preprocessing.
#         If ``num_workers`` = 0, multiprocessing is disabled.
#         Otherwise ``num_workers`` multiprocessing worker is used to process data.
#     pin_memory : boolean, default False
#         If ``True``, the dataloader will copy NDArrays into pinned memory
#         before returning them. Copying from CPU pinned memory to GPU is faster
#         than from normal CPU memory.
#     pin_device_id : int, default 0
#         The device id to use for allocating pinned memory if pin_memory is ``True``
#     prefetch : int, default is `num_workers * 2`
#         The number of prefetching batches only works if `num_workers` > 0.
#         If `prefetch` > 0, it allow worker process to prefetch certain batches before
#         acquiring data from iterators.
#         Note that using large prefetching batch will provide smoother bootstrapping performance,
#         but will consume more shared_memory. Using smaller number may forfeit the purpose of using
#         multiple worker processes, try reduce `num_workers` in this case.
#         By default it defaults to `num_workers * 2`.
#
#     """
#
#     def __init__(self, transform_fns, dataset, interval=1, batch_size=1, shuffle=False,
#                  sampler=None, drop_last=False, batch_sampler=None, collate_fn=default_collate,
#                  num_workers=0, pin_memory=False):
#         super(RandomTransformDataLoader, self).__init__(
#             dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
#             drop_last=drop_last, batch_sampler=batch_sampler, collate_fn=collate_fn,
#             num_workers=num_workers, pin_memory=pin_memory)
#         self._transform_fns = transform_fns
#         assert len(self._transform_fns) > 0
#         self._interval = max(int(interval), 1)
#
#     def __iter__(self):
#         return _MultiDataLoaderIter(self, self._transform_fns, self._interval)
#
#
# if __name__ == '__main__':
#     from torchvision import transforms
#     from data.base import DemoDataset
#     from torch.utils import data
#
#     dataset = DemoDataset(20)
#     transform_fn = [transforms.Compose([transforms.Resize((s, s)), transforms.ToTensor()]) for s in [10, 15, 20, 25,
#                                                                                                      30, 35, 40]]
#     dataset = dataset.transform(transform_fn[0])
#     sampler = data.SequentialSampler(dataset)
#     batch_sampler = data.sampler.BatchSampler(sampler, batch_size=4, drop_last=False)
#     loader = RandomTransformDataLoader(transform_fn, dataset, batch_sampler=batch_sampler, interval=1, num_workers=4)
#     for i, batch in enumerate(loader):
#         print(batch.shape)
