# ~/.bashrc
if [ -f /etc/bashrc ]; then
  . /etc/bashrc
fi

export QT_X11_NO_MITSHM=1

# 默认加载 TED
source /share/project/ianalog/ted_env/ted_bashrc_260130