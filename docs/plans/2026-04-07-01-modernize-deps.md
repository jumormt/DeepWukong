# Plan: Modernize Dependencies

**Epic:** E1: Modernization
**Design:** N/A (straightforward upgrade)

## Summary

Upgrade DeepWukong from 2021-era pinned dependencies to modern compatible versions. The project uses pytorch_lightning 1.3, gensim 3.8, torch-geometric 2.0, and networkx 2.5 — all of which have breaking API changes in their current versions.

**Decisions locked in:**
- Keep the same model architecture and training logic
- Target Python 3.9+ with modern package versions
- Replace `commode_utils` callbacks with inline equivalents if needed

---

## Phase 1: Fix gensim 4.x API changes

### [ ] Task 1.1: Update vocabulary.py
- [ ] In `src/vocabulary.py:33-34`, change `model.vocab` → `model.key_to_index` and `model.vocab[wd].index` → `model.key_to_index[wd]`

### [ ] Task 1.2: Update common_layers.py
- [ ] In `src/models/modules/common_layers.py:127`, change `model.index2word` → `model.index_to_key`

### [ ] Task 1.3: Update word_embedding.py
- [ ] In `src/preprocess/word_embedding.py:75`, change `Word2Vec(size=...)` → `Word2Vec(vector_size=...)`

## Phase 2: Fix pytorch_lightning 2.x API changes

### [ ] Task 2.1: Update train.py
- [ ] Replace `Trainer(gpus=gpu)` with `Trainer(accelerator="auto", devices="auto")`
- [ ] Remove `progress_bar_refresh_rate` (use callback instead or remove)
- [ ] Move `resume_from_checkpoint` to `trainer.fit(ckpt_path=...)`
- [ ] Replace `every_n_val_epochs` with `every_n_epochs` in ModelCheckpoint
- [ ] Replace or remove `commode_utils` callbacks (PrintEpochResultCallback, UploadCheckpointCallback)

### [ ] Task 2.2: Update vd.py
- [ ] Rename `training_epoch_end` → `on_train_epoch_end` (no args in PL 2.x, use self.trainer.callback_metrics or store outputs manually)
- [ ] Rename `validation_epoch_end` → `on_validation_epoch_end`
- [ ] Rename `test_epoch_end` → `on_test_epoch_end`
- [ ] Remove `EPOCH_OUTPUT` type hints
- [ ] Store step outputs manually since PL 2.x doesn't pass them to epoch_end

### [ ] Task 2.3: Update evaluate.py
- [ ] Replace `Trainer(gpus=gpu)` with `Trainer(accelerator="auto", devices="auto")`

### [ ] Task 2.4: Update datamodules.py
- [ ] Fix `transfer_batch_to_device` signature (PL 2.x adds `dataloader_idx` param)

## Phase 3: Fix networkx deprecations

### [ ] Task 3.1: Replace gpickle usage
- [ ] In `src/datas/graphs.py:28`, replace `nx.read_gpickle` with `pickle.load`
- [ ] In `src/data_generator.py`, replace `nx.write_gpickle` with `pickle.dump` and `nx.read_gpickle` with `pickle.load`
- [ ] In `src/preprocess/dataset_generator.py`, same replacements
- [ ] In `src/preprocess/word_embedding.py`, same replacements
- [ ] In `src/utils.py`, same replacements

## Phase 4: Update requirements.txt and test

### [ ] Task 4.1: Write new requirements.txt
- [ ] Remove hard version pins, use compatible ranges
- [ ] Remove torch-scatter and torch-sparse (installed differently now)
- [ ] Update all package names/versions

### [ ] Task 4.2: Install and test imports
- [ ] Install dependencies
- [ ] Test that all modules import without errors

## Phase 5: Update README

### [ ] Task 5.1: Update README.md
- [ ] Update setup instructions for modern environment
- [ ] Note Python version requirements
- [ ] Update any changed commands

## Verification
- [ ] All modules import without errors
- [ ] `PYTHONPATH="." python -c "from src.run import vul_detect"` works
- [ ] PROGRESS.md updated with results
