/**
 * MongoDB Replica Set Initialization Script
 *
 * Run this on the primary node after starting all 3 mongod instances
 * with --replSet "rs0":
 *
 *   mongosh --eval "load('init-replica-set.js')"
 *
 * For a single-node dev replica set (enables change streams & transactions):
 *   mongosh --eval "rs.initiate({_id:'rs0', members:[{_id:0, host:'localhost:27017'}]})"
 */

// 3-node production replica set
rs.initiate({
  _id: "rs0",
  members: [
    { _id: 0, host: "mongo1:27017", priority: 2 },   // preferred primary
    { _id: 1, host: "mongo2:27017", priority: 1 },
    { _id: 2, host: "mongo3:27017", priority: 1 },
  ],
  settings: {
    heartbeatTimeoutSecs: 10,
    electionTimeoutMillis: 10000,
  },
});

// Wait for the primary to be elected
sleep(5000);
print("Replica set status:");
printjson(rs.status());
