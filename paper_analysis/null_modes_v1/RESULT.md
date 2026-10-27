# null_modes_v1 RESULT (post hoc, 2026-10-26; plan sha256 83126af8...)
Using the model's `sigma_star_factor` at each system's frame-0 heavy-atom pose, 7 of the 3,819 corpus systems (0.18%) have more than six discarded modes: 3DRS, 3VTB, 4A7B, 4MT9, 4RVK, 5H8E and 6DTX.
- Each of the seven has a bond angle of 177–180° in frame 0, a near-linear group.
- The earlier 4.7% (7 of 150) came from `exact_sigma_star_noise.py`, which works on ensemble-mean structures with finite-difference rows. That differs by more than a factor of 2, so by the plan's rule the paper now states 0.18%.
- An independent count by T. Ding, 0.195% (12 of 6,144 fold molecules), is consistent with this.
