export NCCL_SOCKET_IFNAME=docker0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

GPU=$1

python -m torch.distributed.launch \
	--nproc_per_node $GPU \
	dist_dataparalle_mail.py --world_size $GPU
