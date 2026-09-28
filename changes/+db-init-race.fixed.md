Opening the database from several threads or processes at once -- as `utrain
tui` does as it starts -- no longer fails with "table runs already exists" on a
new or older database: creating and migrating the schema is done under SQLite's
write lock, once.
