"""
TODO: need to test scaling with
- number of samples evaluated on
- number of processors
- number of data points
"""

# Standard library
from os import path
import time

# Third-party
from astropy.time import Time
import astropy.units as u
import numpy as np
from schwimmbad import choose_pool
from thejoker.data import RVData
from thejoker.log import log as joker_logger
from thejoker.sampler import TheJoker, JokerParams
from twobody import KeplerOrbit

def main(pool):
    seed = 42
    rnd = np.random.RandomState(seed=seed)

    # Generate some fake data (actually from real star...)
    n_total_data = 1024
    EPOCH = Time('J2000') + 172.12*u.day
    orbit = KeplerOrbit(P=50*u.day, e=0.57, omega=2.62*u.rad,
                        M0=1.88*u.rad, t0=EPOCH,
                        i=90*u.deg, Omega=0*u.deg) # these don't matter
    t = Time('J2000') + rnd.uniform(0, 300, n_total_data) * u.day

    rv = 10*u.km/u.s * orbit.unscaled_radial_velocity(t) + 100*u.km/u.s
    err = np.full_like(rv.value, 0.01) * u.km/u.s
    data = RVData(t, rv, stddev=err, t0=EPOCH)

    # -------------------------------------------------------------------------

    params = JokerParams(P_min=1*u.day, P_max=32768*u.day)
    joker = TheJoker(params, random_state=rnd, pool=pool)

    prior_cache_file = '/mnt/home/apricewhelan/projects/hq/cache/P1-32768_prior_samples.hdf5' # TODO

    print("Pool size: {0}".format(pool.size))

    n_iter = 3
    for n_samples in 2 ** np.arange(8, 28+1, 2):
        for n_data in 2 ** np.arange(1, 10+1, 1):
            this_data = data[rnd.choice(n_total_data, n_data, replace=False)]

            dts = []
            for k in range(n_iter):
                t0 = time.time()
                try:
                    samples = joker.rejection_sample(
                        data=this_data, prior_cache_file=prior_cache_file,
                        n_prior_samples=n_samples)

                except Exception as e:
                    print("Failed sampling \n Error: {0}".format(str(e)))
                    continue

                dts.append(time.time() - t0)

            print("{0}, {1}, {1:.3f}, {2:.3f}".format(n_samples, n_data,
                                                      np.mean(dts),
                                                      np.std(dts)))

    pool.close()


if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    args = parser.parse_args()

    for l in [joker_logger]:
        l.setLevel(logging.DEBUG)

    pool_kwargs = dict(mpi=args.mpi, processes=args.n_procs)
    pool = choose_pool(**pool_kwargs)

    main(pool=pool)
