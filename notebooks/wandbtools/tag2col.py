import wandb

if __name__ == "__main__":
    api = wandb.Api()
    wandb.login()

    # select all runs in the project
    project = "team-epoch-iv/detect-sleep-states"
    runs = api.runs(project)

    # go over all runs with a tag, print it, and set the LB score
    for run in runs:
        tags = run.tags
        if len(tags) > 0:
            name = run.name
            if "score" in run.summary:
                score = run.summary["score"]
            else:
                score = "None"
            print(f"{name}: {score} - {tags[0]}")

            # set the LB score for the run from the tag
            run.summary["score_lb"] = float(tags[0])

            # sync the results to the server
            run.update()

