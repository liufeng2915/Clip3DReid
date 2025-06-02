import os
import re
import glob
import h5py
import random
import math
import logging
import numpy as np
import os.path as osp
import re

class Celeb(object):

    def __init__(self, root='', ds_name='', **kwargs):

        if ds_name == 'celeb':
            self.dataset_dir = osp.join(root, 'Celeb-reID')
        elif ds_name == 'celeb_light':
            self.dataset_dir = osp.join(root, 'Celeb-reID-light')
        elif ds_name == 'celeb_blur':
            self.dataset_dir = osp.join(root, 'Celeb-reID-blur')
        elif ds_name == 'celeb_light_blur':
            self.dataset_dir = osp.join(root, 'Celeb-reID-light-blur')

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self._check_before_run()

        train, num_train_pids = self._process_dir(self.train_dir)
        query, num_query_pids  = self._process_dir(self.query_dir)
        gallery, num_gallery_pids = self._process_dir(self.gallery_dir)

        num_total_pids = num_train_pids + num_query_pids + num_gallery_pids
        num_total_imgs = len(train) + len(query) + len(gallery)

        logger = logging.getLogger('reid.dataset')
        logger.info("=> Celib-ReID loaded")
        logger.info("Dataset statistics:")
        logger.info("  --------------------------------------------")
        logger.info("  subset        | # ids | # images")
        logger.info("  ----------------------------------------")
        logger.info("  train         | {:5d} | {:8d}  ".format(num_train_pids, len(train)))
        logger.info("  query         | {:5d} | {:8d}  ".format(num_query_pids, len(query)))
        logger.info("  gallery       | {:5d} | {:8d}  ".format(num_gallery_pids, len(gallery)))
        logger.info("  --------------------------------------------")
        logger.info("  total         | {:5d} | {:8d}  ".format(num_total_pids, num_total_imgs))
        logger.info("  --------------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = 10

        self.pid2clothes = np.ones((self.num_train_pids, self.num_train_clothes))


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, home_dir, relabel=True):
  
        pattern = re.compile(r'([-\d]+)_(\d)')
        fpaths = sorted(glob.glob(osp.join(home_dir, '*.jpg')))
        i = 0
        dataset = []
        pid_container = set()
        all_pids = {}
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue  # junk images are just ignored

            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            cam -= 1
            clothes_id = 1 #-1
            pid = all_pids[pid]
            pid_container.add(pid)
            i = i+1
            dataset.append((fpath, pid, cam, clothes_id))
        num_pids = len(pid_container)

        return dataset, num_pids