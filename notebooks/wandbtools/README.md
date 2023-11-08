# W and B Tools

This is a collection of tools for working with W&B, because we cannot add the Kaggle score to the runs from the web interface.
With the API, the LB score can be added as a proper value we can sort and filter on. Make sure to add the `score_lb` column to the table view to see the result.

Run all the tools directly from the command line

We used to add tags before, run `tag2col` to convert any tag to the `score_lb` column. This should not be needed again.

Run `add_new_score_lb` or to add or update a`score_lb` column for a specific run.
Find the run path in the web interface, from the run overview, this has a button for copying the path.

Use `remove_score_lb` just for mistakes, when accidentally using the wrong run path for instance.

