import wandb

if __name__ == "__main__":
    api = wandb.Api()
    wandb.login()

    # ask for a run path
    run_path = input("Enter run path (copy from run overview): ")

    # get the run
    run = api.run(run_path)
    print(f"Run: {run.name} loaded successfully")

    # ask for a leaderboard score
    score = input("Enter leaderboard score: ")
    score = float(score)

    # set the LB score for the run
    run.summary["score_lb"] = score

    # sync the results to the server
    run.update()
    print("Score updated successfully")
