"""Shadow mode: our own earnings previews, made before each event's knowledge cutoff, plus a record of every live prediction.

- `schedule`: which calendar events are due for a preview (from 12 hours before their cutoff, at most 5 at a time) and claiming them.
- `runner`: generate one cheap preview (`preview_agent`) and store it.
- `store`: one folder per event (own preview and its run record, the competition's preview,
  classified surprises, the feature vector, the prediction), on a Modal Volume in production
  and under `data/surprise/live/` locally.
- `sync`: download the Volume to a local folder and load the records into tables.

At release, `predict()` uses the competition's preview when the event has one, otherwise
our own preview if it finished before the cutoff, otherwise neither.
"""
