import mab_realdata as mr


if __name__ == "__main__":
    num_arms, mincnts = [], []
    for tid, (eid, mid) in mr.dist2tid2eidmid["gauss"].items():
        rd = mr.RealData(eid, mid, "gauss")
        num_arms.append(rd.num_arm)
        mincnts.append(rd.mincnt)
    print("num_arms: ", num_arms)
    print("mincnts: ", mincnts)
    # mean num_arms and mincnts
    print("mean num_arms: ", sum(num_arms) / len(num_arms))
    print("mean mincnts: ", sum(mincnts) / len(mincnts))

