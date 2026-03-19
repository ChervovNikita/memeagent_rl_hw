

from meme_entry import main


if __name__ == "__main__":
    main(env_name="BreakoutDeterministic-v4",
         num_actors=2,
         buffer_size=4e5,
         batch_size=64,
         burnin=0,
         rollout=10
         )

