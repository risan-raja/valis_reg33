#!/bin/bash
cd /storage/valis_reg/

nohup hypercorn --config /storage/valis_reg/image_selector/hypercorn.toml image_selector.app:app &
