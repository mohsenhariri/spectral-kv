PROJECT := $(shell basename $(CURDIR))

BACKUP_DIR := /backup/$(PROJECT)

SYNC_DIR := /storage/sync/git/mohsen/$(PROJECT)


sync-repos:
		if [ ! -d "$(SYNC_DIR)" ]; then mkdir -p $(SYNC_DIR); fi
		rsync -auv --exclude-from=./exclude.lst  . $(SYNC_DIR)/$(PROJECT)

backup:
		if [ ! -d "$(BACKUP_DIR)" ]; then mkdir -p $(BACKUP_DIR); fi
		tar --exclude-from exclude.lst -czvf $(BACKUP_DIR)/$(PROJECT)_$$(date +%Y%m%d_%H-%M-%S).tar.gz ../$(PROJECT)

