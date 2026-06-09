This directory is reserved for local runtime checkpoints that are required to
run `mesh_predict`, but these binary weights are intentionally not committed to
git.

Download the mesh prediction checkpoints from the Google Drive link documented
in `perception_predictor/README.md`, then place them in this directory.

The runtime can also resolve custom checkpoint paths from:

- `MESH_PREDICT_MESH_CKPT`
- `MESH_PREDICT_NKSR_CKPT`
