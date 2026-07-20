"""Model-backed stages (super-resolution, depth, crop segmentation) plus the
shared ``ModelManager`` that lazily loads/unloads them so at most one large
model family is resident in GPU memory at a time.
"""
