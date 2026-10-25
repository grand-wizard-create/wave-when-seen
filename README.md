# Wave when seen

Your arm watches through a camera and waves hello when someone steps in front of it. It waves at
most once every 10 seconds and keeps watching until you stop it.

## What it runs on

- The SO-101 follower arm, or the simulated SO-101 follower.
- A camera, which it needs. Add one to your arm in My robots. A RealSense camera works too.

Every move is slow and smooth: the arm never moves a joint faster than 45 degrees a second, and it
never jumps. Between waves it holds near where it started. The camera pictures stay on this Mac:
the app only looks for change between frames and keeps nothing.

## Run it from Robot App Store

Find Wave when seen in Discover, install it, choose your arm, and click Run. Make sure the arm has
room to rise in front of it, then step in front of the camera.

## Run it from a terminal

```
monorobot pull grand-wizard-create/wave-when-seen
monorobot robot bind hf_so101_follower --kwarg port=<port>
monorobot start wave-when-seen
```

Replace `<port>` with your arm's USB port.

## Stop it

In Robot App Store, press Space or Esc. In a terminal, press Ctrl+C: the arm folds back to rest.
Press Ctrl+C a second time to stop the arm where it is.

## License

Apache License 2.0. See `LICENSE`.
