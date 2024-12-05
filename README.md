This Psychopy experiment is a human replication of the mouse visual decision-making task developed by IBL.
Contact Anne Urai, a.e.urai@fsw.leidenuniv.nl / see [this doc](https://docs.google.com/document/d/1C6Kt_tYg0wLJQ1GE0N0mQVeitvk-i0vjs0vuYjYIJsQ/edit?tab=t.0) for instructions on running the task.

The most recent and up to date version is `WIN_cursor_ibl.psyexp`. This version was designed to run on Windows using Psychopy v.2024.1.4. The experiment can be opened and edited on later versions and on Mac, but note that on a Mac, the mouse component will not allow to click "continue" buttons due to the retina screen doubling the mouse coordinates.

---

The items relevant for running the current task are:
- “pregenerated_sequences” = folder where original mouse IBL session information was taken and used to create session information for the current version. Also includes two python scripts to do this. Not necessary to run the task, but useful if you want to change session info like block length and stimulus presentation order.
- “pregen_sequence_0.xlsx” … through _9 = the excel files that defines the predetermined sequences determining target eccentricity, contrast, and ITI. This info is used in the trial loop. The last digit of participant ID determined which pregen_sequence file is used.
- “2000.wav” = sound file for correct response
- “whitenoise.wav” = sound file for wrong response
- “example left.png” = example screenshot of task stimuli to show during instructions, where target is on the left
- “example right.png” = example screenshot of task stimuli to show during instructions, where target is on the right
- “fixation_object.png” = image of the fixation cross
- “readme.md” = README file
- “WIN_cursor_ibl.psyexp” = the experiment file
- “WIN_cursor_ibl_lastrun.py” = automatically generated python code showing the last execution of the .psyexp file
