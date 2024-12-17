import os, re, sys, time
from subprocess import PIPE, Popen


class WindowGetter:
    def get_activityname(self):

        root = Popen( ['xprop', '-root', '_NET_ACTIVE_WINDOW'], stdout = PIPE )
        stdout, stderr = root.communicate()
        m = re.search( b'^_NET_ACTIVE_WINDOW.* ([\w]+)$', stdout )

        if m is not None:

            window_id   = m.group( 1 )

            windowname  = None
            window = Popen( ['xprop', '-id', window_id, 'WM_NAME'], stdout = PIPE )
            stdout, stderr = window.communicate()
            wmatch = re.match( b'WM_NAME\(\w+\) = (?P<name>.+)$', stdout )
            if wmatch is not None:
                windowname = wmatch.group( 'name' ).decode( 'UTF-8' ).strip( '"' )

            processname1, processname2 = None, None
            process = Popen( ['xprop', '-id', window_id, 'WM_CLASS'], stdout = PIPE )
            stdout, stderr = process.communicate()
            pmatch = re.match( b'WM_CLASS\(\w+\) = (?P<name>.+)$', stdout )
            if pmatch is not None:
                processname1, processname2 = pmatch.group( 'name' ).decode( 'UTF-8' ).split( ', ' )
                processname1 = processname1.strip( '"' )
                processname2 = processname2.strip( '"' )

            return {
                'windowname':   windowname,
                'processname1': processname1,
                'processname2': processname2
                }

        return {
            'windowname':   None,
            'processname1': None,
            'processname2': None
            }
    