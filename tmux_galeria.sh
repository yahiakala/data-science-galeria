# --------------------------------------------------------- 
# TMUX DATA SCIENCE FOR DATA-SCIENCE-GALERIA
# ---------------------------------------------------------
# Create a tmux session as an IDE for Data Science Projects
# Created with Python in mind.
# ---------------------------------------------------------
# REQUIREMENTS
# ---------------------------------------------------------
# tmux > 2
# conda
# vim
# ---------------------------------------------------------
# USAGE
# ---------------------------------------------------------
# sh ~/path/to/tmux_datascience.sh <project-name>
# ---------------------------------------------------------

SESSIONNAME='dsg'
PROJNAME=$1
CONDANAME='ds-galeria' # conda env name.

# first arg is the name of session, second is name of first window
tmux new-session -s $SESSIONNAME -n bashwindow -d
tmux send-keys "cd $PROJNAME" ENTER

# Add new window
tmux new-window -n 3WAY

# Main code editor

# Console
tmux split-window -h
tmux send-keys "conda activate $CONDANAME" ENTER
tmux send-keys "clear" ENTER

# Create a third window for JLAB
tmux new-window -n JLB
tmux send-keys "conda activate $CONDANAME" ENTER
tmux send-keys "clear" ENTER

# Attach to session.
tmux attach-session -t $SESSIONNAME
