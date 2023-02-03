import numpy as np
import math
import random
import sys
from datetime import datetime
from numpy.random import normal
from scipy.stats import norm
from scipy.optimize import fsolve
import argparse
from dataclasses import dataclass

import os
import pandas as pd
from meta_dict import dist2tid2eidmid, dist2absdiffstats

class RealData:
    def __init__(self, eid, mid):
        #
        self.eid, self.mid = eid, mid
        #
        if eid == 0 or mid == 0:
            self.valid = False
            return
        self.valid = True

        fname = f"{eid}_{mid}.csv"
        check_tid = os.getenv("TASK_INDEX")
        if check_tid is None:
            fname = f"~/Downloads/kdd2023_mab/real_data/{fname}"
        else:
            pass
        self.df = pd.read_csv(fname)
        tmp = self.df.groupby(["exptid", "groupid"]).count().reset_index()
        #
        self.gids, self.gid2uv= [], {}
        for i, row in tmp.iterrows():
            self.gids.append(row.groupid)
            self.gid2uv[str(row.groupid)] = row.uin
        sgids = sorted(self.gids)
        #
        self.control_gids, self.treat_gids = sgids[:2], sgids[2:]
        #
        self.armid2gids = {}
        self.armid2gids["0"] = self.control_gids
        for i, gid in enumerate(self.treat_gids):
            self.armid2gids[str(i + 1)] = [gid]
        #
        self.gid2armid = dict()
        for gid in self.control_gids:
            self.gid2armid[str(gid)] = 0
        for i, gid in enumerate(sorted(self.treat_gids)):
            self.gid2armid[str(gid)] = i + 1

        self.num_arm = 1 + len(self.treat_gids)
        self.df["armid"] = self.df['groupid'].apply(
                lambda x: self.gid2armid[str(x)])
        self.armid2df, self.armid2nextidx = dict(), dict()
        for armid, df in self.df.groupby("armid"):
            self.armid2df[str(armid)] = df
            self.armid2nextidx[str(armid)] = 0

        self.control_mean = self.armid2df["0"]["numerator"].mean()

        self.armid2mean, self.armid2var, self.armid2cnt = dict(), dict(), dict()
        for armid, df in self.armid2df.items():
            self.armid2mean[armid] = df["numerator"].mean()
            self.armid2var[armid] = df["numerator"].var(ddof=1)
            self.armid2cnt[armid] = df["numerator"].count()
        self.mincnt = min(self.armid2cnt.values())
        self.means = list(self.armid2mean.values())
        self.vars = list(self.armid2var.values())

        self.gaps = []
        for i in range(len(self.means)):
            for j in range(i + 1, len(self.means)):
                self.gaps.append(abs(self.means[i] - self.means[j]))
        # self.gap_mean = np.mean(self.gaps)
        # self.gap_var = np.var(self.gaps)

# dist2eidmid = {
#     "bernoulli": dist2tid2eidmid["bernoulli"].get(str(tid), (0, 0)),
#     "gauss": dist2tid2eidmid["gauss"].get(str(tid), (0, 0)),
# }

def get_dist2gaps():
    dist2gaps = dict()
    for dist, tid2eidmid in dist2tid2eidmid.items():
        dist2gaps[dist] = list()
        for tid, (eid, mid) in tid2eidmid.items():
            rd = RealData(eid, mid)
            if not rd.valid:
                continue
            print(dist, tid, eid, mid, rd.gaps)
            dist2gaps[dist].extend(rd.gaps)
    for dist, gaps in dist2gaps.items():
        print(dist, np.mean(gaps), np.var(gaps, ddof=1))
    # print(dist2gaps)

def get_mid2gaps():
    mid2gaps = dict()
    for dist, tid2eidmid in dist2tid2eidmid.items():
        for tid, (eid, mid) in tid2eidmid.items():
            if mid not in mid2gaps:
                mid2gaps[mid] = list()
            rd = RealData(eid, mid)
            if not rd.valid:
                continue
            print(dist, tid, eid, mid, rd.gaps)
            mid2gaps[mid].extend(rd.gaps)
    for mid, gaps in mid2gaps.items():
        # print(mid, np.mean(gaps), np.var(gaps, ddof=1))
        print(f"{str(mid)}: [{np.mean(gaps)}, {np.var(gaps, ddof=1)}],")
    # print(dist2gaps)


if __name__ == "__main__":
    # get_dist2gaps()
    get_mid2gaps()
