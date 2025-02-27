g-init:
		touch .gitignore
		git init
		git add .
		git commit -m "initial commit"

g-commit: format pylint-dev
		git commit -m "$(filter-out $@,$(MAKECMDGOALS))"

g-log:
		git log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit

pre-add: sort format clean-command