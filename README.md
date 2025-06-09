# Novelty and Resonance Detection Using Word Embeddings

This is the repository for our paper using NLP methods to detect novel and resonant ideas in text. The repository is managed by [Hannah Bailey](https://github.com/hannahlsbailey) and Tom Nicholls.

The code files are run in this order:
1. clearning.ipynb (as of now only run on the "Who Leads Who Follows Paper" section)
2. pre_training_cleaning_removing_links_and_rts.py
3. checked_cleaned_df.py
4. training_model.py (I ran three versions of training_model - each using a different training method. You can have a look at the others (training_model_pos_pairs.py and training_model_simcse.py if you want, but I decided that the method used in training_model.py worked best, so this is what we will continue to use here for the package).
5. generating_embeddings.py (again, there are some redundant scripts here, including generating_embeddings_debug.py, generating_embeddings_pospairs.py, generating_cosine_similarity_scores_for_manual_validation and generating_embeddings_simcse.py). Although, on second thought, it might be a good idea to incorporate some of the output produced from generating_cosine_similarity_scores_for_manual_validation.py into the package, so users can check that their results are valid.
6. Perhaps add - print function to print out some similar/less similar examples as a sanity check. Perhaps build into validate_embeddings.py - this step involves some manual "threshold setting".
8. combine_df_and_embs.py (perhaps worth integrating this into the previous script? This script combines the df and embeddings into a single df, which I found useful to avoid merging issues in the next steps)
9. splittling_df_embs_for_ntr.py (this splits the combined df into shards - probably unnecessary for this package, but including it for reference here)
10. novelty_transience_resonance_sharded_quicker.py (v important script - calculates the resonance and impact scores for each document)
11. ntr_single_df.py (again, probably not relevant here - this merges the shards)
12. ntr_analysis_pt1_summary_stats_w_significance.py (summary statistics for novelty, transience and resonance)
13. ntr_analysis_pt2_example_post.py (prints most impactful posts)
14. ntr_analusis_pt3_who_leads.py (produces heatmap)
15. ntr_analysis_pt4_bubble_plot.py (produces bubble plot)
