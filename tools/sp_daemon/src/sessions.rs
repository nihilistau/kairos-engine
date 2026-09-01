use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Registry of active chat sessions. Keyed by the u64 chat_id returned by register().
pub struct Sessions {
    inner: Mutex<HashMap<u64, Arc<AtomicI32>>>,
    next_id: AtomicU64,
}

impl Sessions {
    pub fn new() -> Arc<Self> {
        Arc::new(Sessions {
            inner: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        })
    }

    /// Insert a new chat with its cancel_flag; returns the assigned chat_id.
    pub fn register(&self, flag: Arc<AtomicI32>) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.inner.lock().unwrap().insert(id, flag);
        id
    }

    /// Flip the cancel_flag for chat_id. Returns false if id not found.
    pub fn abort(&self, id: u64) -> bool {
        let guard = self.inner.lock().unwrap();
        if let Some(flag) = guard.get(&id) {
            flag.store(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Remove a completed or aborted chat from the table.
    pub fn remove(&self, id: u64) {
        self.inner.lock().unwrap().remove(&id);
    }
}
