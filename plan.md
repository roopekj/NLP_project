# Project Plan

- [ ] Finnish 🇫🇮 basic frontend
  - [ ] Node positions (Merlo)
  - [x] Proper labelling according to clusters (Sergio)
  - [x] Different "zoom" levels -> slider (Sergio)
  - [ ] NEW: Edit notebook to do the same steps but iterating over all zoom levels (i.e., number of clusters) and create all data json files (for samples, labels and clusters) (Someone please)
  - [ ] (Optional priority) Click on node to see article preview and cluster label (Merlo)
  - [ ] (Optional) Select different algos/clusterings (Anyone who wants to give it a shot?)
  - [ ] (Optional) Tweak color scheme (Anyone who wants to give it a shot?)
  - [ ] (Optional) Connect similar nodes manually/automagically (Anyone who wants to give it a shot?)

- [ ] Additional backend features
  - [ ] Better labeling of clusters (Anyone who wants to give it a shot?)
  - [ ] Hierarchical clustering (for zoom levels?) (Merlo)
  - [ ] (Optional) More jsons with different cluster methods, embeddings etc. (Roope)

- [ ] Project report (together after the project is finnished 🇫🇮)

## Labelling

Right now, the labels for each node are the labels we have produced using tSNE for its cluster. It would be better to display both a label for its cluster (a few-words summary of the union of the documents in the cluster) and also a label for the document (a few-words summary of the document).
