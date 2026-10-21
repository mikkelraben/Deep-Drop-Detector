# Deep Drop Detector (3D)
## What Is A Drop?
A drop is defined as a point in a song where the music reaches a climax (higher intensity) introducing or reintroducing many elements/sounds (most importantly the bass). This usually follow a build-up section, which has less intensity and fewer layers of elements/sounds.

## How Is The Data Labeled?
In each song, every drop is labelled. The timestamp of the drop to a precision of 0.1 seconds is written down in a csv file. The csv file has the following format:

```csv
filename;timestamp
```

If a song has multiple drops, each timestamp will be seperated by a comma. For example, if a song has 3 drops at 1.2 seconds, 2.5 seconds, and 3.7 seconds, the csv file will look like this:

```csv
filename;timestamp
song1.wav;1.2,2.5,3.7
```
