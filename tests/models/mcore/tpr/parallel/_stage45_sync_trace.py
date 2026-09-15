"""Dependency-free synchronized boundary logger; diagnostic runs only."""

from contextlib import contextmanager


class SyncTrace:
    def __init__(self, synchronize, emit, label):
        self.synchronize = synchronize
        self.emit = emit
        self.label = label
        self.stack = []
        self.failed = False

    @contextmanager
    def span(self, name):
        self.stack.append(name)
        boundary = f"STAGE45-DIAG {self.label} path={'/'.join(self.stack)}"
        phase = "pre_sync"
        try:
            self.emit(f"{boundary} PRE_SYNC")
            self.synchronize()
            self.emit(f"{boundary} BEGIN")
            phase = "body"
            yield
            phase = "post_sync"
            self.emit(f"{boundary} POST_SYNC")
            self.synchronize()
            self.emit(f"{boundary} END")
        except BaseException as exc:
            if not self.failed:
                self.failed = True
                self.emit(f"{boundary} FIRST_ERROR phase={phase} type={type(exc).__name__}")
            raise
        finally:
            # Never submit a device operation on exceptional unwind.
            self.stack.pop()

    def install(self, patch, executor_type, autograd):
        for method in ("push", "visit_leaf", "pop", "_forward", "_compute_loss"):
            original = getattr(executor_type, method)

            def wrapped(executor, segment, *args, original=original, method=method, **kwargs):
                spec = executor.plan.get(segment) if isinstance(segment, int) else segment
                info = (f"{method}(sid={spec.segment_id},T={spec.length},P={spec.prefix_length}"
                        f",no_grad={kwargs.get('no_grad', 'NA')})")
                with self.span(info):
                    return original(executor, segment, *args, **kwargs)

            patch.setattr(executor_type, method, wrapped)
        original_backward = autograd.backward

        def backward(*args, **kwargs):
            with self.span("backward"):
                return original_backward(*args, **kwargs)

        patch.setattr(autograd, "backward", backward)
