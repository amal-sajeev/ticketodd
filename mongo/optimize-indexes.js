/**
 * MongoDB Index Optimization Script
 *
 * Run after initial deployment to add compound indexes for common query patterns.
 * These complement the basic indexes created by the app on startup.
 *
 * Usage:
 *   mongosh grievance_system optimize-indexes.js
 */

const db = db.getSiblingDB("grievance_system");

// ---------------------------------------------------------------------------
// Grievances — compound indexes for common filter+sort patterns
// ---------------------------------------------------------------------------

// Officer dashboard: filter by status + department, sort by priority
db.grievances.createIndex(
  { status: 1, department: 1, priority: -1, created_at: -1 },
  { name: "idx_status_dept_priority_created", background: true }
);

// Citizen view: filter by citizen_user_id + status, sort by created_at
db.grievances.createIndex(
  { citizen_user_id: 1, status: 1, created_at: -1 },
  { name: "idx_citizen_status_created", background: true }
);

// SLA monitoring: unresolved + sla_deadline
db.grievances.createIndex(
  { status: 1, sla_deadline: 1 },
  { name: "idx_status_sla", background: true,
    partialFilterExpression: { status: { $in: ["pending", "in_progress", "escalated"] } } }
);

// Analytics: department + created_at for time-series aggregations
db.grievances.createIndex(
  { department: 1, created_at: -1 },
  { name: "idx_dept_created", background: true }
);

// Public feed: is_public + status + created_at
db.grievances.createIndex(
  { is_public: 1, status: 1, created_at: -1 },
  { name: "idx_public_status_created", background: true }
);

// Scheme-matched filter
db.grievances.createIndex(
  { "scheme_match": 1, status: 1 },
  { name: "idx_scheme_match_status", background: true,
    partialFilterExpression: { "scheme_match": { $exists: true, $ne: null } } }
);

// Multi-department sub-tasks
db.grievances.createIndex(
  { "sub_tasks": 1, created_at: -1 },
  { name: "idx_subtasks_created", background: true,
    partialFilterExpression: { "sub_tasks.0": { $exists: true } } }
);

// Impact score sorting
db.grievances.createIndex(
  { impact_score: -1 },
  { name: "idx_impact_score", background: true }
);

// Assigned officer workload
db.grievances.createIndex(
  { assigned_officer: 1, status: 1 },
  { name: "idx_officer_status", background: true }
);

// ---------------------------------------------------------------------------
// Spam tracking
// ---------------------------------------------------------------------------
db.spam_tracking.createIndex(
  { is_blocked: 1 },
  { name: "idx_blocked", background: true }
);

db.spam_tracking.createIndex(
  { "duplicate_hashes.hash": 1 },
  { name: "idx_dup_hash", background: true }
);

db.spam_tracking.createIndex(
  { ip_addresses: 1 },
  { name: "idx_ip_addresses", background: true }
);

// ---------------------------------------------------------------------------
// Vouches
// ---------------------------------------------------------------------------
db.vouches.createIndex(
  { grievance_id: 1, created_at: -1 },
  { name: "idx_vouch_grievance_created", background: true }
);

// ---------------------------------------------------------------------------
// Systemic issues
// ---------------------------------------------------------------------------
db.systemic_issues.createIndex(
  { department: 1, status: 1, created_at: -1 },
  { name: "idx_systemic_dept_status", background: true }
);

// ---------------------------------------------------------------------------
// Forecasts
// ---------------------------------------------------------------------------
db.forecasts.createIndex(
  { forecast_date: -1 },
  { name: "idx_forecast_date", background: true }
);

print("Index optimization complete. Current indexes:");
printjson(db.grievances.getIndexes());
