current_dir = $(shell pwd)

.PHONY: style-fix
style-fix:
	isort ACGPN_inference ACGPN_train scripts
	autoflake --remove-all-unused-imports --recursive --in-place --ignore-init-module-imports \
		 ACGPN_inference ACGPN_train scripts
	black -l 120 -t py37 \
		--exclude .*\/\(migrations\)\/.* \
		ACGPN_inference ACGPN_train scripts

.PHONY: style-check
style-check:
	isort --check-only ACGPN_inference ACGPN_train scripts
	flake8 ACGPN_inference ACGPN_train scripts
	black -l 120 -t py37 --fast --check \
		--exclude .*\/\(migrations\)\/.* \
		ACGPN_inference ACGPN_train scripts
