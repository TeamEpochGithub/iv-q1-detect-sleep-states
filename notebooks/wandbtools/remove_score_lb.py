import wandb

if __name__ == "__main__":
    api = wandb.Api()
    wandb.login()

    # ask for a run path
    run_path = input("Enter run path (copy from run overview): ")

    # get the run
    run = api.run(run_path)
    print(f"Run: {run.name} loaded successfully")

    # remove the score
    run.summary["score_lb"] = None

    # sync the results to the server
    run.update()
    print("Score removed successfully")
