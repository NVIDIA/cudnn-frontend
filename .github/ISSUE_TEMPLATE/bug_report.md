---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**Expected behavior**
A clear and concise description of what you expected to happen.

**System Environment (please complete the following information):**
 - cudnn_frontend version: [e.g. v1.4.0]
 - cudnn_backend version: [e.g. v9.1.0]
 - GPU arch: [e.g. RTX 4090]
 - cuda runtime version: [e.g. 12.4]
 - cuda driver version: [e.g. 553.04]
 - host compiler: [e.g. clang19]
 - OS: [e.g. ubuntu22.04]

**API logs**
Please attach API logs for both cudnn_frontend and cudnn_backend.
```
// For cudnn_frontend
export CUDNN_FRONTEND_LOG_FLIE=fe.log
export CUDNN_FRONTEND_LOG_INFO=1

// For cudnn_backend
export CUDNN_LOGLEVEL_DBG=3
export CUDNN_LOGDEST_DBG=be.log
```

**To Reproduce**
Steps to reproduce the behavior:
1. '...'
2. '....'
3. '....'

**Additional context**
Add any other context about the problem here.
