#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on oktober 25, 2024, at 16:00
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

import psychopy
psychopy.useVersion('2024.1.5')


# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# This section of the EyeLink Initialize component code imports some
# modules we need, manages data filenames, allows for dummy mode configuration
# (for testing experiments without an eye tracker), connects to the tracker,
# and defines some helper functions (which can be called later)
import pylink
import time
import platform
from PIL import Image  # for preparing the Host backdrop image
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from string import ascii_letters, digits
from psychopy import gui

import psychopy_eyelink
print('EyeLink Plugin For PsychoPy Version = ' + str(psychopy_eyelink.__version__))

script_path = os.path.dirname(sys.argv[0])
if len(script_path) != 0:
    os.chdir(script_path)

# Set this variable to True if you use the built-in retina screen as your
# primary display device on macOS. If have an external monitor, set this
# variable True if you choose to "Optimize for Built-in Retina Display"
# in the Displays preference settings.
use_retina = False

# Set this variable to True to run the script in "Dummy Mode"
dummy_mode = False

# Prompt user to specify an EDF data filename
# before we open a fullscreen window
dlg_title = "Enter EDF Filename"
dlg_prompt = "Please enter a file name with 8 or fewer characters [letters, numbers, and underscore]."
# loop until we get a valid filename
while True:
    dlg = gui.Dlg(dlg_title)
    dlg.addText(dlg_prompt)
    dlg.addField("Filename", "EDF Filename:","Test")
    # show dialog and wait for OK or Cancel
    ok_data = dlg.show()
    if dlg.OK:  # if ok_data is not None
        print("EDF data filename: {}".format(ok_data["Filename"]))
    else:
        print("user cancelled")
        core.quit()
        sys.exit()

    # get the string entered by the experimenter
    tmp_str = ok_data["Filename"]
    # strip trailing characters, ignore the ".edf" extension
    edf_fname = tmp_str.rstrip().split(".")[0]

    # check if the filename is valid (length <= 8 & no special char)
    allowed_char = ascii_letters + digits + "_"
    if not all([c in allowed_char for c in edf_fname]):
        print("ERROR: Invalid EDF filename")
    elif len(edf_fname) > 8:
        print("ERROR: EDF filename should not exceed 8 characters")
    else:
        break# Set up a folder to store the EDF data files and the associated resources
# e.g., files defining the interest areas used in each trial
results_folder = "results"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# We download EDF data file from the EyeLink Host PC to the local hard
# drive at the end of each testing session, here we rename the EDF to
# include session start date/time
time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
session_identifier = edf_fname + time_str

# create a folder for the current testing session in the "results" folder
session_folder = os.path.join(results_folder, session_identifier)
if not os.path.exists(session_folder):
    os.makedirs(session_folder)

# For macOS users check if they have a retina screen
if 'Darwin' in platform.system():
    dlg = gui.Dlg("Retina Screen?")
    dlg.addText("What type of screen will the experiment run on?")
    dlg.addField("Screen Type", choices=["High Resolution (Retina, 2k, 4k, 5k)", "Standard Resolution (HD or lower)"])
    # show dialog and wait for OK or Cancel
    ok_data = dlg.show()
    if dlg.OK:
        if dlg.data["Screen Type"] == "High Resolution (Retina, 2k, 4k, 5k)":  
            use_retina = True
        else:
            use_retina = False
    else:
        print('user cancelled')
        core.quit()
        sys.exit()

# Connect to the EyeLink Host PC
# The Host IP address, by default, is "100.1.1.1".
# the "el_tracker" objected created here can be accessed through the Pylink
# Set the Host PC address to "None" (without quotes) to run the script
# in "Dummy Mode"
if dummy_mode:
    el_tracker = pylink.EyeLink(None)
else:
    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        dlg = gui.Dlg("Dummy Mode?")
        dlg.addText("Could not connect to tracker at 100.1.1.1 -- continue in Dummy Mode?")
        # show dialog and wait for OK or Cancel
        ok_data = dlg.show()
        if dlg.OK:  # if ok_data is not None
            dummy_mode = True
            el_tracker = pylink.EyeLink(None)
        else:
            print("user cancelled")
            core.quit()
            sys.exit()

eyelinkThisFrameCallOnFlipScheduled = False
eyelinkLastFlipTime = 0.0
zeroTimeIAS = 0.0
zeroTimeDLF = 0.0
sentIASFileMessage = False
sentDrawListMessage = False

def clear_screen(win,genv):
    """ clear up the PsychoPy window"""
    win.fillColor = genv.getBackgroundColor()
    win.flip()

def show_msg(win, genv, text, wait_for_keypress=True):
    """ Show task instructions on screen"""
    scn_width, scn_height = win.size
    msg = visual.TextStim(win, text,
                          color=genv.getForegroundColor(),
                          wrapWidth=scn_width/2)
    clear_screen(win,genv)
    msg.draw()
    win.flip()

    # wait indefinitely, terminates upon any key press
    if wait_for_keypress:
        kb = keyboard.Keyboard()
        waitKeys = kb.waitKeys(keyList=None, waitRelease=True, clear=True)
        clear_screen(win,genv)

def terminate_task(win,genv,edf_file,session_folder,session_identifier):
    """ Terminate the task gracefully and retrieve the EDF data file
    """
    el_tracker = pylink.getEYELINK()

    if el_tracker.isConnected():
        # Terminate the current trial first if the task terminated prematurely
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            abort_trial(win,genv)

        # Put tracker in Offline mode
        el_tracker.setOfflineMode()

        # Clear the Host PC screen and wait for 500 ms
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)

        # Close the edf data file on the Host
        el_tracker.closeDataFile()

        # Show a file transfer message on the screen
        msg = 'EDF data is transferring from EyeLink Host PC...'
        show_msg(win, genv, msg, wait_for_keypress=False)

        # Download the EDF data file from the Host PC to a local data folder
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        local_edf = os.path.join(session_folder, session_identifier + '.EDF')
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

        # Close the link to the tracker.
        el_tracker.close()

    # close the PsychoPy window
    win.close()

    # quit PsychoPy
    core.quit()
    sys.exit()

def abort_trial(win,genv):
    """Ends recording """
    el_tracker = pylink.getEYELINK()

    # Stop recording
    if el_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        el_tracker.stopRecording()

    # clear the screen
    clear_screen(win,genv)
    # Send a message to clear the Data Viewer screen
    bgcolor_RGB = (116, 116, 116)
    el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # send a message to mark trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)
    return pylink.TRIAL_ERROR

# this method converts PsychoPy position values to EyeLink position values
# EyeLink position values are in pixel units and are such that 0,0 corresponds 
# to the top-left corner of the screen and increase as position moves right/down
def eyelink_pos(pos,winSize):
    screenUnitType = 'pix'
    scn_width,scn_height = winSize
    if screenUnitType == 'pix':
        elPos = [pos[0] + scn_width/2,scn_height/2 - pos[1]]
    elif screenUnitType == 'height':
        elPos = [scn_width/2 + pos[0] * scn_height,scn_height/2 + pos[1] * scn_height]
    elif screenUnitType == "norm":
        elPos = [(scn_width/2 * pos[0]) + scn_width/2,scn_height/2 + pos[1] * scn_height]
    else:
        print("ERROR:  Only pixel, height, and norm units supported for conversion to EyeLink position units")
    return [int(round(elPos[0])),int(round(elPos[1]))]

# this method converts PsychoPy size values to EyeLink size values
# EyeLink size values are in pixels
def eyelink_size(size,winSize):
    screenUnitType = 'pix'
    scn_width,scn_height = winSize
    if len(size) == 1:
        size = [size[0],size[0]]
    if screenUnitType == 'pix':
        elSize = [size[0],size[1]]
    elif screenUnitType == 'height':
        elSize = [int(round(scn_height*size[0])),int(round(scn_height*size[1]))]
    elif screenUnitType == "norm":
        elSize = [size[0]/2 * scn_width,size[1]/2 * scn_height]
    else:
        print("ERROR:  Only pixel, height, and norm units supported for conversion to EyeLink position units")
    return [int(round(elSize[0])),int(round(elSize[1]))]

# this method converts PsychoPy color values to EyeLink color values
def eyelink_color(color):
    elColor = (int(round((win.color[0]+1)/2*255)),int(round((win.color[1]+1)/2*255)),int(round((win.color[2]+1)/2*255)))
    return elColor


# This method, created by the EyeLink MarkEventsTrial component code, will get called to handle
# sending event marking messages, logging Data Viewer (DV) stimulus drawing info, logging DV interest area info,
# sending DV Target Position Messages, and/or logging DV video frame marking info=information
def eyelink_onFlip_MarkEventsTrial(globalClock,win,scn_width,scn_height,allStimComponentsForEyeLinkMonitoring,\
    componentsForEyeLinkStimEventMessages):
    global eyelinkThisFrameCallOnFlipScheduled,eyelinkLastFlipTime,zeroTimeDLF,sentDrawListMessage
    # Log the time of the current frame onset for upcoming messaging/event logging
    currentFrameTime = float(globalClock.getTime())

    # Go through all stimulus components that need to be checked (for event marking,
    # DV drawing, and/or interest area logging) to see if any have just ONSET
    for thisComponent in allStimComponentsForEyeLinkMonitoring:
        # Check if the component has just onset
        if thisComponent.tStartRefresh is not None and not thisComponent.elOnsetDetected:
            # Check whether we need to mark stimulus onset (and log a trial variable logging this time) for the component
            if thisComponent in componentsForEyeLinkStimEventMessages:
                el_tracker.sendMessage('%s %s_ONSET' % (int(round((globalClock.getTime()-thisComponent.tStartRefresh)*1000)),thisComponent.name))
                el_tracker.sendMessage('!V TRIAL_VAR %s_ONSET_TIME %i' % (thisComponent.name,thisComponent.tStartRefresh*1000))
                # Convert the component's position to EyeLink units and log this value under .elPos
                # Also create lastelPos/lastelSize to store pos/size of the previous position, which is needed for IAS file writing
                thisComponent.elPos = eyelink_pos(thisComponent.pos,[scn_width,scn_height])
                thisComponent.elSize = eyelink_size(thisComponent.size,[scn_width,scn_height])
                thisComponent.lastelPos = thisComponent.elPos
                thisComponent.lastelSize = thisComponent.elSize

            thisComponent.elOnsetDetected = True

            # update the position (in EyeLink coordinates) for upcoming usage
            thisComponent.lastelPos = thisComponent.elPos
            thisComponent.lastelSize = thisComponent.elSize
    # Go through all stimulus components that need to be checked (for event marking,
    # DV drawing, and/or interest area logging) to see if any have just OFFSET
    for thisComponent in allStimComponentsForEyeLinkMonitoring:
        # Check if the component has just offset
        if thisComponent.tStopRefresh is not None and thisComponent.tStartRefresh is not None and \
            not thisComponent.elOffsetDetected:
            # send a message marking that component's offset in the EDF
            if thisComponent in componentsForEyeLinkStimEventMessages:
                el_tracker.sendMessage('%s %s_OFFSET' % (int(round((globalClock.getTime()-thisComponent.tStopRefresh)*1000)),thisComponent.name))
            thisComponent.elOffsetDetected = True 
    # This logs whether a call to this method has already been scheduled for the upcoming retrace
    # And is used to help ensure we schedule only one callOnFlip call for each retrace
    eyelinkThisFrameCallOnFlipScheduled = False
    # This stores the time of the last retrace and can be used in Code components to 
    # check the time of the previous screen flip
    eyelinkLastFlipTime = float(currentFrameTime)
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.5'
expName = 'eyetrack_ibl'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
_loggingLevel = logging.getLevel('exp')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, data.getDateStr(format="%Y_%b_%d_%H.%M.%S"))
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\ExperimentData\\human_IBL EyeLink testing\\WIN_cursor_ibl_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=True,
            monitor='labMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=False,
            units='pix', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'pix'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = True
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # create speaker 'sound_trial_start_3'
    deviceManager.addDevice(
        deviceName='sound_trial_start_3',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=7.0
    )
    # create speaker 'sound_no_resp_3'
    deviceManager.addDevice(
        deviceName='sound_no_resp_3',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=7.0
    )
    # create speaker 'feedback_sound_2'
    deviceManager.addDevice(
        deviceName='feedback_sound_2',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=7.0
    )
    # create speaker 'sound_trial_start'
    deviceManager.addDevice(
        deviceName='sound_trial_start',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=7.0
    )
    # create speaker 'sound_no_resp'
    deviceManager.addDevice(
        deviceName='sound_no_resp',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=7.0
    )
    # create speaker 'feedback_sound'
    deviceManager.addDevice(
        deviceName='feedback_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=7.0
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome" ---
    welcome_position = visual.TextStim(win=win, name='welcome_position',
        text="Welcome to the experiment! \n\nPlease keep your head on the chinrest in front of you.\n\nMake sure that you can comfortably reach the mouse; you will not need the keyboard for now.\n\nIt is really important that you stay in the same position throughout the experiment.\n\nClick 'Continue' when you are ready.",
        font='Arial',
        units='height', pos=(0, 0.05), height=0.035, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    continue_txt = visual.TextStim(win=win, name='continue_txt',
        text='Continue',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouse_1 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_1.mouseClock = core.Clock()
    # This section of the EyeLink Initialize component code opens an EDF file,
    # writes some header text to the file, and configures some tracker settings
    el_tracker = pylink.getEYELINK()
    global edf_fname
    # Open an EDF data file on the Host PC
    edf_file = edf_fname + ".EDF"
    try:
        el_tracker.openDataFile(edf_file)
    except RuntimeError as err:
        print("ERROR:", err)
        # close the link if we have one open
        if el_tracker.isConnected():
            el_tracker.close()
        core.quit()
        sys.exit()
    
    # Add a header text to the EDF file to identify the current experiment name
    # This is OPTIONAL. If your text starts with "RECORDED BY " it will be
    # available in DataViewer's Inspector window by clicking
    # the EDF session node in the top panel and looking for the "Recorded By:"
    # field in the bottom panel of the Inspector.
    preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
    el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)
    
    # Configure the tracker
    #
    # Put the tracker in offline mode before we change tracking parameters
    el_tracker.setOfflineMode()
    
    # Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000,
    # 5-EyeLink 1000 Plus, 6-Portable DUO
    eyelink_ver = 0  # set version to 0, in case running in Dummy mode
    if not dummy_mode:
        vstr = el_tracker.getTrackerVersionString()
        eyelink_ver = int(vstr.split()[-1].split(".")[0])
        # print out some version info in the shell
        print("Running experiment on %s, version %d" % (vstr, eyelink_ver))
    
    # File and Link data control
    # what eye events to save in the EDF file, include everything by default
    file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
    # what eye events to make available over the link, include everything by default
    link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
    # what sample data to save in the EDF data file and to make available
    # over the link, include the 'HTARGET' flag to save head target sticker
    # data for supported eye trackers
    if eyelink_ver > 3:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
    else:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
    el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
    el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
    el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
    el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)
    # Set a gamepad button to accept calibration/drift check target
    # You need a supported gamepad/button box that is connected to the Host PC
    el_tracker.sendCommand("button_function 5 'accept_target_fixation'")
    
    global eyelinkThisFrameCallOnFlipScheduled,eyelinkLastFlipTime,zeroTimeDLF,sentDrawListMessage,zeroTimeIAS,sentIASFileMessage
    
    # --- Initialize components for Routine "setup_camera" ---
    camera_info_txt = visual.TextStim(win=win, name='camera_info_txt',
        text="We will now set up the eye-tracking camera.\nThis will take just a couple minutes.\n\nA dot will appear on the screen at different locations.\nSimply move your eyes to follow the dot on the screen.\nPlease keep staring directly at the dot until it moves to a new location.\n\nClick 'Continue' to start the camera setup.",
        font='Arial',
        units='height', pos=(0, 0.05), height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    continue_txt_17 = visual.TextStim(win=win, name='continue_txt_17',
        text='Continue',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouse_19 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_19.mouseClock = core.Clock()
    Initialize = event.Mouse(win=win)
    CameraSetup = event.Mouse(win=win)
    
    # --- Initialize components for Routine "el_start_rec" ---
    HostDrawing = event.Mouse(win=win)
    StartRecord = event.Mouse(win=win)
    
    # --- Initialize components for Routine "demographics" ---
    # Run 'Begin Experiment' code from mouse_visible_4
    tot_points = 0
    demogr_txt = visual.TextStim(win=win, name='demogr_txt',
        text='Thank you! The camera setup is done.\n\nWe will now ask you some information about your age, gender, and handedness.',
        font='Arial',
        units='height', pos=(0, 0), height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    continue_txt_8 = visual.TextStim(win=win, name='continue_txt_8',
        text='Continue',
        font='Arial',
        units='height', pos=(0, -0.3), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mouse_10 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_10.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "age" ---
    age_txt = visual.TextStim(win=win, name='age_txt',
        text="How old are you?\n\nClick on a number to select your age, then click 'Continue'. \nIf you prefer not to say, click 'Continue'.",
        font='Arial',
        units='deg', pos=(0, 5), height=0.75, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    age_slider = visual.Slider(win=win, name='age_slider',
        startValue=None, size=(35, 2), pos=(0, 0), units='deg',
        labels=[18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], ticks=(18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40), granularity=1.0,
        style='choice', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Arial', labelHeight=0.75,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    continue_txt_4 = visual.TextStim(win=win, name='continue_txt_4',
        text='Continue',
        font='Arial',
        units='deg', pos=(0, -5), height=0.75, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mouse_4 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_4.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "gender" ---
    gender_txt = visual.TextStim(win=win, name='gender_txt',
        text='What is your gender?',
        font='Arial',
        units='height', pos=(0, 0.1), height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    woman = visual.TextStim(win=win, name='woman',
        text='Woman',
        font='Arial',
        units='height', pos=(-0.4, -0.1), height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    man = visual.TextStim(win=win, name='man',
        text='Man',
        font='Arial',
        units='height', pos=(-0.15, -0.1), height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    nonbinary = visual.TextStim(win=win, name='nonbinary',
        text='Non-binary',
        font='Arial',
        units='height', pos=(0.09, -0.1), height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    other = visual.TextStim(win=win, name='other',
        text='Other/\nPrefer not to say',
        font='Arial',
        units='height', pos=(0.4, -0.1), height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    mouse_5 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_5.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "handedness" ---
    handedness_txt = visual.TextStim(win=win, name='handedness_txt',
        text='Are you:',
        font='Arial',
        units='height', pos=(0, 0.1), height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    right_hand = visual.TextStim(win=win, name='right_hand',
        text='Right-handed',
        font='Arial',
        units='height', pos=(-0.4, -0.1), height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    left_hand = visual.TextStim(win=win, name='left_hand',
        text='Left-handed',
        font='Arial',
        units='height', pos=(0, -0.1), height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    ambidx = visual.TextStim(win=win, name='ambidx',
        text='Other/\nPrefer not to say',
        font='Arial',
        units='height', pos=(0.4, -0.1), height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    mouse_6 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_6.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "short_instructions" ---
    # Run 'Begin Experiment' code from short_instr
    #if int(expInfo['participant']) % 4 in [1, 2]:
    #    instructions = 1
    #else:
    #    instructions = 0
    
    instructions = 0
    thisExp.addData("instructions", instructions)
    short_instructions_txt = visual.TextStim(win=win, name='short_instructions_txt',
        text="Thank you! We will now move on to the experiment.\n\nYou will take part in a computer game.\nHowever, you will not receive any instructions: the goal is to figure out the rules of the game by yourself.\n\nTo interact with the game, you will only need to move the mouse; clicking does not do anything. \n\nImportant: during the game, you will see a red cross in the middle of the screen. Please try to keep your eyes fixated on this cross throughout the experiment.\n\nIf you have any questions, feel free to ask out loud, The experimenter can hear you and will help you out.\n\nWhen you are ready, click 'Start experiment'.",
        font='Arial',
        units='height', pos=(0, 0.1), height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    continue_txt_21 = visual.TextStim(win=win, name='continue_txt_21',
        text='Start experiment',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    mouse_25 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_25.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "pre_instr" ---
    pre_instr_txt = visual.TextStim(win=win, name='pre_instr_txt',
        text="Thank you! We will now move on to the experiment.\n\nYou will take part in a computer game.\n\nYou will now see some written instructions, as well as two example images of what the stimuli look like.\n\nYou will then practice the game in a few example trials.\n\n\nClick 'Continue' to see the instructions.",
        font='Arial',
        units='height', pos=(0, 0.1), height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    continue_txt_18 = visual.TextStim(win=win, name='continue_txt_18',
        text='Continue',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mouse_22 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_22.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "instructions_1" ---
    # Run 'Begin Experiment' code from save_winsize
    thisExp.addData("win_size", win.size)
    instruction_txt = visual.TextStim(win=win, name='instruction_txt',
        text="The game consists of many trials. In each trial, you will see a red fixation point in the middle of the screen. \n\nAt the beginning of each trial you will hear a sharp beep. \nYou will then see two target images appear on each side of the screen. \n\nClick 'Continue' to see more instructions and examples.",
        font='Arial',
        units='height', pos=(0, 0.1), height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    continue_txt_2 = visual.TextStim(win=win, name='continue_txt_2',
        text='Continue',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    mouse_2 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_2.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "instructions_2" ---
    example_r = visual.ImageStim(
        win=win,
        name='example_r', units='height', 
        image='example right.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), size=(1.2, 0.6),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    instruction_txt_2 = visual.TextStim(win=win, name='instruction_txt_2',
        text="On each trial, you must decide which of the two target images has the higher contrast (i.e., appears darker). \nTo select the target with the higher contrast, simply move the mouse until the chosen target reaches the center of the screen.\n\nFor example, in the image below, the right target has the higher contrast, so you would move your mouse to the left.\n\nClick 'Continue' to see more instructions and another example.",
        font='Arial',
        units='height', pos=(0, 0.3), height=0.03, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    continue_txt_9 = visual.TextStim(win=win, name='continue_txt_9',
        text='Continue',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    mouse_11 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_11.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "instructions_3" ---
    example_l = visual.ImageStim(
        win=win,
        name='example_l', units='height', 
        image='example left.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), size=(1.2, 0.6),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    instruction_txt_3 = visual.TextStim(win=win, name='instruction_txt_3',
        text="In this example image, the left target has the higher contrast, so you would move your mouse to the right.\n\nTo interact with the game, you will only need to move the mouse; clicking does not do anything.\n\nClick 'Continue' to see more instructions.",
        font='Arial',
        units='height', pos=(0, 0.3), height=0.03, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    continue_txt_10 = visual.TextStim(win=win, name='continue_txt_10',
        text='Continue',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    mouse_12 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_12.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "instructions_4" ---
    instruction_txt_4 = visual.TextStim(win=win, name='instruction_txt_4',
        text="You will hear a high beep if you answer correctly.\nIf you answer incorrectly, you will hear a buzzing noise instead.\nIf you take too long to answer, you will hear a low beep.\n\nPlease try to be as accurate and fast as possible, and try not to let the trial time-out without a response.\n\n\nClick 'Continue' to practice a few example trials.",
        font='Arial',
        units='height', pos=(0, 0), height=0.035, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    continue_txt_11 = visual.TextStim(win=win, name='continue_txt_11',
        text='Continue',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mouse_13 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_13.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "fix_practice" ---
    # Run 'Begin Experiment' code from set_contrast_side_2
    import random
    fixation_5 = visual.ImageStim(
        win=win,
        name='fixation_5', units='deg', 
        image='fixation_object.png', mask='circle', anchor='center',
        ori=0.0, pos=(0, 0), size=1.0,
        color='white', colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    
    # --- Initialize components for Routine "trial_practice" ---
    # Run 'Begin Experiment' code from dragging_code_3
    correct = 9
    dot_3 = visual.ShapeStim(
        win=win, name='dot_3',units='pix', 
        size=(5, 5), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[0,0,0], fillColor=[0,0,0],
        opacity=None, depth=-2.0, interpolate=True)
    fixation_6 = visual.ImageStim(
        win=win,
        name='fixation_6', units='deg', 
        image='fixation_object.png', mask='circle', anchor='center',
        ori=0.0, pos=(0, 0), size=1.0,
        color='white', colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    grating_l_3 = visual.GratingStim(
        win=win, name='grating_l_3',units='pix', 
        tex='sin', mask='gauss', anchor='center',
        ori=0.0, pos=[0,0], size=(618, 618), sf=0.0073, phase=1.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-4.0)
    grating_r_3 = visual.GratingStim(
        win=win, name='grating_r_3',units='pix', 
        tex='sin', mask='gauss', anchor='center',
        ori=0.0, pos=[0,0], size=(618, 618), sf=0.0073, phase=1.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-5.0)
    mouse_21 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_21.mouseClock = core.Clock()
    sound_trial_start_3 = sound.Sound(
        'A', 
        secs=0.1, 
        stereo=True, 
        hamming=True, 
        speaker='sound_trial_start_3',    name='sound_trial_start_3'
    )
    sound_trial_start_3.setVolume(0.1)
    sound_no_resp_3 = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_no_resp_3',    name='sound_no_resp_3'
    )
    sound_no_resp_3.setVolume(0.1)
    
    # --- Initialize components for Routine "feedback_practice" ---
    # Run 'Begin Experiment' code from feedback_code_2
    fb_sound = 100
    fb_dur = 10
    fb_sound_dur = 5
    fb_volume = 0.1
    feedback_sound_2 = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='feedback_sound_2',    name='feedback_sound_2'
    )
    feedback_sound_2.setVolume(1.0)
    fixation_7 = visual.ImageStim(
        win=win,
        name='fixation_7', units='deg', 
        image='fixation_object.png', mask='circle', anchor='center',
        ori=0.0, pos=(0, 0), size=(0.75, 0.75),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    
    # --- Initialize components for Routine "start_session" ---
    end_practice_txt = visual.TextStim(win=win, name='end_practice_txt',
        text="That is the end of the practice trials.\n\nHere is a last reminder of the instructions:\n\nIn each trial, you will hear a beep and two targets will appear.\nYou must decide which target has the higher contrast.\nTo respond, move your mouse left or right to bring the chosen target into the middle of the screen. You don't need to click, simply drag the mouse.\n\nIf you are correct, you will hear a high beep; if you are incorrect, you will hear a buzzing noise.\n\nImportant: durig the game you will see a red cross i the middle of the screen. Please try to keep your eyes fixated on this cross throughout the experiment.\n\nIf you have any questions, feel free to ask out loud, the experimenter can hear you and will help you out.\n\nYou will now start the real experiment. Good luck!",
        font='Arial',
        units='height', pos=(0, 0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    continue_txt_12 = visual.TextStim(win=win, name='continue_txt_12',
        text='Start experiment',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    mouse_14 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_14.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "fix" ---
    # Run 'Begin Experiment' code from set_iti
    #iti = 0
    #maxDur = 0.7
    #minDur = 0.4
    fixation_2 = visual.ImageStim(
        win=win,
        name='fixation_2', units='deg', 
        image='fixation_object.png', mask='circle', anchor='center',
        ori=0.0, pos=(0, 0), size=1.0,
        color='white', colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    
    # --- Initialize components for Routine "trial" ---
    # Run 'Begin Experiment' code from dragging_code
    correct = 9
    
    dot = visual.ShapeStim(
        win=win, name='dot',units='pix', 
        size=(5, 5), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[0,0,0], fillColor=[0,0,0],
        opacity=None, depth=-1.0, interpolate=True)
    grating_l = visual.GratingStim(
        win=win, name='grating_l',units='pix', 
        tex='sin', mask='gauss', anchor='center',
        ori=0.0, pos=[0,0], size=(618, 618), sf=0.0073, phase=1.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-2.0)
    grating_r = visual.GratingStim(
        win=win, name='grating_r',units='pix', 
        tex='sin', mask='gauss', anchor='center',
        ori=0.0, pos=[0,0], size=(618, 618), sf=0.0073, phase=1.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-3.0)
    fixation = visual.ImageStim(
        win=win,
        name='fixation', units='deg', 
        image='fixation_object.png', mask='circle', anchor='center',
        ori=0.0, pos=(0, 0), size=1.0,
        color='white', colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    sound_trial_start = sound.Sound(
        'A', 
        secs=0.1, 
        stereo=True, 
        hamming=True, 
        speaker='sound_trial_start',    name='sound_trial_start'
    )
    sound_trial_start.setVolume(0.1)
    sound_no_resp = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker='sound_no_resp',    name='sound_no_resp'
    )
    sound_no_resp.setVolume(0.1)
    MarkEventsTrial = event.Mouse(win=win)
    
    # --- Initialize components for Routine "feedback" ---
    # Run 'Begin Experiment' code from feedback_code
    fb_sound = 100
    fb_dur = 10
    fb_sound_dur = 5
    fb_volume = 0.1
    feedback_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='feedback_sound',    name='feedback_sound'
    )
    feedback_sound.setVolume(1.0)
    fixation_3 = visual.ImageStim(
        win=win,
        name='fixation_3', units='deg', 
        image='fixation_object.png', mask='circle', anchor='center',
        ori=0.0, pos=(0, 0), size=(0.75, 0.75),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "record_delay" ---
    blank_txt = visual.TextStim(win=win, name='blank_txt',
        text='wait...',
        font='Arial',
        units='height', pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color=[0.0000, 0.0000, 0.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "el_stop_rec" ---
    StopRecord = event.Mouse(win=win)
    
    # --- Initialize components for Routine "end_task" ---
    end_task_txt = visual.TextStim(win=win, name='end_task_txt',
        text="That's the end of the game.\nThank you for participating! \n\nBefore we show you your final score, we will ask you a few questions about the game you just completed.\n\nYou no longer need to sit still or keep your head on the chin rest. \nPlease take the keyboard you see on the desk, and use it to type your answers. \n\nWhen answering questions, please do not press 'Enter' on the keyboard. Simply use punctuation to separate your sentences.\n\nClick 'Continue' to see the first question.",
        font='Arial',
        units='height', pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    continue_txt_22 = visual.TextStim(win=win, name='continue_txt_22',
        text='Continue',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    mouse_26 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_26.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "question1" ---
    q1_txt = visual.TextStim(win=win, name='q1_txt',
        text="Please describe how you think the game worked\nand what you think the rules were.\n\nClick 'Continue' to see the next question.",
        font='Arial',
        units='height', pos=(0, 0.4), height=0.03, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    textbox1 = visual.TextBox2(
         win, text=None, placeholder='Type your answer...', font='Arial',
         pos=(0, 0),units='height',     letterHeight=0.03,
         size=(0.8, 0.5), borderWidth=0.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.02, alignment='top-left',
         anchor='center', overflow='scroll',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox1',
         depth=-1, autoLog=True,
    )
    continue_txt_23 = visual.TextStim(win=win, name='continue_txt_23',
        text='Continue',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mouse_27 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_27.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "question2" ---
    q2_txt = visual.TextStim(win=win, name='q2_txt',
        text="How well do you think you did in the game?\n\nClick 'Continue' to see the next question.",
        font='Arial',
        units='height', pos=(0, 0.4), height=0.03, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    textbox2 = visual.TextBox2(
         win, text=None, placeholder='Type your answer...', font='Arial',
         pos=(0, 0),units='height',     letterHeight=0.03,
         size=(0.8, 0.5), borderWidth=0.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.02, alignment='top-left',
         anchor='center', overflow='scroll',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox2',
         depth=-1, autoLog=True,
    )
    continue_txt_24 = visual.TextStim(win=win, name='continue_txt_24',
        text='Continue',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mouse_28 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_28.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "question3" ---
    q3_txt = visual.TextStim(win=win, name='q3_txt',
        text="What strategies did you use during the game?\n\nClick 'Continue' to see the next question.",
        font='Arial',
        units='height', pos=(0, 0.4), height=0.03, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    textbox3 = visual.TextBox2(
         win, text=None, placeholder='Type your answer...', font='Arial',
         pos=(0, 0),units='height',     letterHeight=0.03,
         size=(0.8, 0.5), borderWidth=0.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.02, alignment='top-left',
         anchor='center', overflow='scroll',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox3',
         depth=-1, autoLog=True,
    )
    continue_txt_25 = visual.TextStim(win=win, name='continue_txt_25',
        text='Continue',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mouse_29 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_29.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "question4" ---
    q4_txt = visual.TextStim(win=win, name='q4_txt',
        text="Did you notice any patterns in the game?\n\nClick 'Continue' to see the next question.",
        font='Arial',
        units='height', pos=(0, 0.4), height=0.03, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    textbox4 = visual.TextBox2(
         win, text=None, placeholder='Type your answer...', font='Arial',
         pos=(0, 0),units='height',     letterHeight=0.03,
         size=(0.8, 0.5), borderWidth=0.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.02, alignment='top-left',
         anchor='center', overflow='scroll',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox4',
         depth=-1, autoLog=True,
    )
    continue_txt_26 = visual.TextStim(win=win, name='continue_txt_26',
        text='Continue',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mouse_30 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_30.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "question5" ---
    q5_txt = visual.TextStim(win=win, name='q5_txt',
        text="Is there anything else you would like to add?\n\nClick 'Continue' when you are done.",
        font='Arial',
        units='height', pos=(0, 0.4), height=0.03, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    textbox5 = visual.TextBox2(
         win, text=None, placeholder='Type your answer...', font='Arial',
         pos=(0, 0),units='height',     letterHeight=0.03,
         size=(0.8, 0.5), borderWidth=0.0,
         color='black', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.02, alignment='top-left',
         anchor='center', overflow='scroll',
         fillColor=None, borderColor='black',
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox5',
         depth=-1, autoLog=True,
    )
    continue_txt_29 = visual.TextStim(win=win, name='continue_txt_29',
        text='Continue',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    mouse_33 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_33.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "end_exp" ---
    end_exp_txt = visual.TextStim(win=win, name='end_exp_txt',
        text='',
        font='Arial',
        units='height', pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    continue_txt_27 = visual.TextStim(win=win, name='continue_txt_27',
        text='Finish',
        font='Arial',
        units='height', pos=(0, -0.4), height=0.04, wrapWidth=None, ori=0.0, 
        color='orange', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouse_31 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_31.mouseClock = core.Clock()
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('welcome.started', globalClock.getTime(format='float'))
    # setup some python lists for storing info about the mouse_1
    mouse_1.x = []
    mouse_1.y = []
    mouse_1.leftButton = []
    mouse_1.midButton = []
    mouse_1.rightButton = []
    mouse_1.time = []
    mouse_1.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    welcomeComponents = [welcome_position, continue_txt, mouse_1]
    for thisComponent in welcomeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_position* updates
        
        # if welcome_position is starting this frame...
        if welcome_position.status == NOT_STARTED and frameN >= 0.0:
            # keep track of start time/frame for later
            welcome_position.frameNStart = frameN  # exact frame index
            welcome_position.tStart = t  # local t and not account for scr refresh
            welcome_position.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_position, 'tStartRefresh')  # time at next scr refresh
            # update status
            welcome_position.status = STARTED
            welcome_position.setAutoDraw(True)
        
        # if welcome_position is active this frame...
        if welcome_position.status == STARTED:
            # update params
            pass
        
        # *continue_txt* updates
        
        # if continue_txt is starting this frame...
        if continue_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt.frameNStart = frameN  # exact frame index
            continue_txt.tStart = t  # local t and not account for scr refresh
            continue_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt.started')
            # update status
            continue_txt.status = STARTED
            continue_txt.setAutoDraw(True)
        
        # if continue_txt is active this frame...
        if continue_txt.status == STARTED:
            # update params
            pass
        # *mouse_1* updates
        
        # if mouse_1 is starting this frame...
        if mouse_1.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_1.frameNStart = frameN  # exact frame index
            mouse_1.tStart = t  # local t and not account for scr refresh
            mouse_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_1, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_1.status = STARTED
            mouse_1.mouseClock.reset()
            prevButtonState = mouse_1.getPressed()  # if button is down already this ISN'T a new click
        if mouse_1.status == STARTED:  # only update if started and not finished!
            buttons = mouse_1.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_1):
                            gotValidClick = True
                            mouse_1.clicked_name.append(obj.name)
                    x, y = mouse_1.getPos()
                    mouse_1.x.append(x)
                    mouse_1.y.append(y)
                    buttons = mouse_1.getPressed()
                    mouse_1.leftButton.append(buttons[0])
                    mouse_1.midButton.append(buttons[1])
                    mouse_1.rightButton.append(buttons[2])
                    mouse_1.time.append(mouse_1.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('welcome.stopped', globalClock.getTime(format='float'))
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_1.x', mouse_1.x)
    thisExp.addData('mouse_1.y', mouse_1.y)
    thisExp.addData('mouse_1.leftButton', mouse_1.leftButton)
    thisExp.addData('mouse_1.midButton', mouse_1.midButton)
    thisExp.addData('mouse_1.rightButton', mouse_1.rightButton)
    thisExp.addData('mouse_1.time', mouse_1.time)
    thisExp.addData('mouse_1.clicked_name', mouse_1.clicked_name)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "setup_camera" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('setup_camera.started', globalClock.getTime(format='float'))
    # setup some python lists for storing info about the mouse_19
    mouse_19.x = []
    mouse_19.y = []
    mouse_19.leftButton = []
    mouse_19.midButton = []
    mouse_19.rightButton = []
    mouse_19.time = []
    mouse_19.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    setup_cameraComponents = [camera_info_txt, continue_txt_17, mouse_19, Initialize, CameraSetup]
    for thisComponent in setup_cameraComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "setup_camera" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *camera_info_txt* updates
        
        # if camera_info_txt is starting this frame...
        if camera_info_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            camera_info_txt.frameNStart = frameN  # exact frame index
            camera_info_txt.tStart = t  # local t and not account for scr refresh
            camera_info_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(camera_info_txt, 'tStartRefresh')  # time at next scr refresh
            # update status
            camera_info_txt.status = STARTED
            camera_info_txt.setAutoDraw(True)
        
        # if camera_info_txt is active this frame...
        if camera_info_txt.status == STARTED:
            # update params
            pass
        
        # *continue_txt_17* updates
        
        # if continue_txt_17 is starting this frame...
        if continue_txt_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_17.frameNStart = frameN  # exact frame index
            continue_txt_17.tStart = t  # local t and not account for scr refresh
            continue_txt_17.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_17, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_17.started')
            # update status
            continue_txt_17.status = STARTED
            continue_txt_17.setAutoDraw(True)
        
        # if continue_txt_17 is active this frame...
        if continue_txt_17.status == STARTED:
            # update params
            pass
        # *mouse_19* updates
        
        # if mouse_19 is starting this frame...
        if mouse_19.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_19.frameNStart = frameN  # exact frame index
            mouse_19.tStart = t  # local t and not account for scr refresh
            mouse_19.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_19, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_19.status = STARTED
            mouse_19.mouseClock.reset()
            prevButtonState = mouse_19.getPressed()  # if button is down already this ISN'T a new click
        if mouse_19.status == STARTED:  # only update if started and not finished!
            buttons = mouse_19.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_17, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_19):
                            gotValidClick = True
                            mouse_19.clicked_name.append(obj.name)
                    x, y = mouse_19.getPos()
                    mouse_19.x.append(x)
                    mouse_19.y.append(y)
                    buttons = mouse_19.getPressed()
                    mouse_19.leftButton.append(buttons[0])
                    mouse_19.midButton.append(buttons[1])
                    mouse_19.rightButton.append(buttons[2])
                    mouse_19.time.append(mouse_19.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in setup_cameraComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "setup_camera" ---
    for thisComponent in setup_cameraComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('setup_camera.stopped', globalClock.getTime(format='float'))
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_19.x', mouse_19.x)
    thisExp.addData('mouse_19.y', mouse_19.y)
    thisExp.addData('mouse_19.leftButton', mouse_19.leftButton)
    thisExp.addData('mouse_19.midButton', mouse_19.midButton)
    thisExp.addData('mouse_19.rightButton', mouse_19.rightButton)
    thisExp.addData('mouse_19.time', mouse_19.time)
    thisExp.addData('mouse_19.clicked_name', mouse_19.clicked_name)
    # This section of the EyeLink Initialize component code gets graphic 
    # information from Psychopy, sets the screen_pixel_coords on the Host PC based
    # on these values, and logs the screen resolution for Data Viewer via 
    # a DISPLAY_COORDS message
    
    # get the native screen resolution used by PsychoPy
    scn_width, scn_height = win.size
    # resolution fix for Mac retina displays
    if 'Darwin' in platform.system():
        if use_retina:
            scn_width = int(scn_width/2.0)
            scn_height = int(scn_height/2.0)
    
    # Pass the display pixel coordinates (left, top, right, bottom) to the tracker
    # see the EyeLink Installation Guide, "Customizing Screen Settings"
    el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendCommand(el_coords)
    
    # Write a DISPLAY_COORDS message to the EDF file
    # Data Viewer needs this piece of info for proper visualization, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendMessage(dv_coords)# This Begin Experiment tab of the elTrial component just initializes
    # a trial counter variable at the beginning of the experiment
    trial_index = 1
    # Configure a graphics environment (genv) for tracker calibration
    genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win, True)
    print(genv)  # print out the version number of the CoreGraphics library
    
    # resolution fix for macOS retina display issues
    if use_retina:
        genv.fixMacRetinaDisplay()
    # Request Pylink to use the PsychoPy window we opened above for calibration
    pylink.openGraphicsEx(genv)
    # Create an array of pixels to assist in transferring content to the Host PC backdrop
    rgbBGColor = eyelink_color(win.color)
    blankHostPixels = [[rgbBGColor for i in range(scn_width)]
        for j in range(scn_height)]
    # This section of EyeLink CameraSetup component code configures some
    # graphics options for calibration, and then performs a camera setup
    # so that you can set up the eye tracker and calibrate/validate the participant
    # graphics options for calibration, and then performs a camera setup
    # so that you can set up the eye tracker and calibrate/validate the participant
    
    # Set background and foreground colors for the calibration target
    # in PsychoPy, (-1, -1, -1)=black, (1, 1, 1)=white, (0, 0, 0)=mid-gray
    background_color = win.color
    foreground_color = (1,1,1)
    genv.setCalibrationColors(foreground_color, background_color)
    
    # Set up the calibration/validation target
    #
    # The target could be a "circle" (default), a "picture", a "movie" clip,
    # or a rotating "spiral". To configure the type of drift check target, set
    # genv.setTargetType to "circle", "picture", "movie", or "spiral", e.g.,
    genv.setTargetType('circle')
    #
    genv.setTargetSize(24)
    
    # Beeps to play during calibration, validation and drift correction
    # parameters: target, good, error
    #     target -- sound to play when target moves
    #     good -- sound to play on successful operation
    #     error -- sound to play on failure or interruption
    # Each parameter could be ''--default sound, 'off'--no sound, or a wav file
    genv.setCalibrationSounds('', '', '')
    
    # Choose a calibration type, H3, HV3, HV5, HV13 (HV = horizontal/vertical),
    el_tracker.sendCommand("calibration_type = " 'HV9')
    #clear the screen before we begin Camera Setup mode
    clear_screen(win,genv)
    
    
    # Go into Camera Setup mode so that participant setup/EyeLink calibration/validation can be performed
    # skip this step if running the script in Dummy Mode
    if not dummy_mode:
        try:
            el_tracker.doTrackerSetup()
        except RuntimeError as err:
            print('ERROR:', err)
            el_tracker.exitCalibration()
        else:
            win.mouseVisible = False
    thisExp.nextEntry()
    # the Routine "setup_camera" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "el_start_rec" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('el_start_rec.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    el_start_recComponents = [HostDrawing, StartRecord]
    for thisComponent in el_start_recComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "el_start_rec" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.001:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in el_start_recComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "el_start_rec" ---
    for thisComponent in el_start_recComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('el_start_rec.stopped', globalClock.getTime(format='float'))
    # This section of EyeLink HostDrawing component code provides options for sending images/shapes
    # representing stimuli to the Host PC backdrop for real-time gaze monitoring
    
    # get a reference to the currently active EyeLink connection
    el_tracker = pylink.getEYELINK()
    # put the tracker in the offline mode first
    el_tracker.setOfflineMode()
    # clear the host screen before we draw the backdrop
    el_tracker.sendCommand('clear_screen 0')
    # Draw rectangles along the edges of components to serve as landmarks on the Host PC backdrop during recording
    # For a list of supported draw commands, see the "COMMANDS.INI" file on the Host PC
    componentDrawListForHostBackdrop = [dot, fixation]
    for thisComponent in componentDrawListForHostBackdrop:
            thisComponent.elPos = eyelink_pos(thisComponent.pos,[scn_width,scn_height])
            thisComponent.elSize = eyelink_size(thisComponent.size,[scn_width,scn_height])
            drawColor = 4
            drawCommand = "draw_box = %i %i %i %i %i" % (thisComponent.elPos[0] - thisComponent.elSize[0]/2,
                thisComponent.elPos[1] - thisComponent.elSize[1]/2, thisComponent.elPos[0] + thisComponent.elSize[0]/2,
                thisComponent.elPos[1] + thisComponent.elSize[1]/2, drawColor)
            el_tracker.sendCommand(drawCommand)
    # record_status_message -- send a messgae string to the Host PC that will be present during recording
    el_tracker.sendCommand("record_status_message '%s'" % ("Block recording"))
    # This section of EyeLink StartRecord component code starts eye tracker recording,
    # sends a trial start (i.e., TRIALID) message to the EDF, 
    # and logs which eye is tracked
    
    # get a reference to the currently active EyeLink connection
    el_tracker = pylink.getEYELINK()
    # Send a "TRIALID" message to mark the start of a trial, see the following Data Viewer User Manual:
    # "Protocol for EyeLink Data to Viewer Integration -> Defining the Start and End of a Trial"
    el_tracker.sendMessage('TRIALID %d' % trial_index)
    # Log the trial index at the start of recording in case there will be multiple trials within one recording
    trialIDAtRecordingStart = int(trial_index)
    # Log the routine index at the start of recording in case there will be multiple routines within one recording
    routine_index = 1
    # put tracker in idle/offline mode before recording
    el_tracker.setOfflineMode()
    # Start recording, logging all samples/events to the EDF and making all data available over the link
    # arguments: sample_to_file, events_to_file, sample_over_link, events_over_link (1-yes, 0-no)
    try:
        el_tracker.startRecording(1, 1, 1, 1)
    except RuntimeError as error:
        print("ERROR:", error)
        abort_trial(genv)
    # Allocate some time for the tracker to cache some samples before allowing
    # trial stimulus presentation to proceed
    pylink.pumpDelay(100)
    # determine which eye(s) is/are available
    # 0-left, 1-right, 2-binocular
    eye_used = el_tracker.eyeAvailable()
    if eye_used == 1:
        el_tracker.sendMessage("EYE_USED 1 RIGHT")
    elif eye_used == 0 or eye_used == 2:
        el_tracker.sendMessage("EYE_USED 0 LEFT")
        eye_used = 0
    else:
        print("ERROR: Could not get eye information!")
    #routineForceEnded = True
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.001000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "demographics" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('demographics.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from mouse_visible_4
    win.mouseVisible = True
    # setup some python lists for storing info about the mouse_10
    mouse_10.x = []
    mouse_10.y = []
    mouse_10.leftButton = []
    mouse_10.midButton = []
    mouse_10.rightButton = []
    mouse_10.time = []
    mouse_10.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    demographicsComponents = [demogr_txt, continue_txt_8, mouse_10]
    for thisComponent in demographicsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "demographics" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *demogr_txt* updates
        
        # if demogr_txt is starting this frame...
        if demogr_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            demogr_txt.frameNStart = frameN  # exact frame index
            demogr_txt.tStart = t  # local t and not account for scr refresh
            demogr_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(demogr_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'demogr_txt.started')
            # update status
            demogr_txt.status = STARTED
            demogr_txt.setAutoDraw(True)
        
        # if demogr_txt is active this frame...
        if demogr_txt.status == STARTED:
            # update params
            pass
        
        # *continue_txt_8* updates
        
        # if continue_txt_8 is starting this frame...
        if continue_txt_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_8.frameNStart = frameN  # exact frame index
            continue_txt_8.tStart = t  # local t and not account for scr refresh
            continue_txt_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_8.started')
            # update status
            continue_txt_8.status = STARTED
            continue_txt_8.setAutoDraw(True)
        
        # if continue_txt_8 is active this frame...
        if continue_txt_8.status == STARTED:
            # update params
            pass
        # *mouse_10* updates
        
        # if mouse_10 is starting this frame...
        if mouse_10.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_10.frameNStart = frameN  # exact frame index
            mouse_10.tStart = t  # local t and not account for scr refresh
            mouse_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_10, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_10.status = STARTED
            mouse_10.mouseClock.reset()
            prevButtonState = mouse_10.getPressed()  # if button is down already this ISN'T a new click
        if mouse_10.status == STARTED:  # only update if started and not finished!
            buttons = mouse_10.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_8, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_10):
                            gotValidClick = True
                            mouse_10.clicked_name.append(obj.name)
                    x, y = mouse_10.getPos()
                    mouse_10.x.append(x)
                    mouse_10.y.append(y)
                    buttons = mouse_10.getPressed()
                    mouse_10.leftButton.append(buttons[0])
                    mouse_10.midButton.append(buttons[1])
                    mouse_10.rightButton.append(buttons[2])
                    mouse_10.time.append(mouse_10.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in demographicsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "demographics" ---
    for thisComponent in demographicsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('demographics.stopped', globalClock.getTime(format='float'))
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_10.x', mouse_10.x)
    thisExp.addData('mouse_10.y', mouse_10.y)
    thisExp.addData('mouse_10.leftButton', mouse_10.leftButton)
    thisExp.addData('mouse_10.midButton', mouse_10.midButton)
    thisExp.addData('mouse_10.rightButton', mouse_10.rightButton)
    thisExp.addData('mouse_10.time', mouse_10.time)
    thisExp.addData('mouse_10.clicked_name', mouse_10.clicked_name)
    thisExp.nextEntry()
    # the Routine "demographics" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "age" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('age.started', globalClock.getTime(format='float'))
    age_slider.reset()
    # setup some python lists for storing info about the mouse_4
    mouse_4.x = []
    mouse_4.y = []
    mouse_4.leftButton = []
    mouse_4.midButton = []
    mouse_4.rightButton = []
    mouse_4.time = []
    mouse_4.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    ageComponents = [age_txt, age_slider, continue_txt_4, mouse_4]
    for thisComponent in ageComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "age" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *age_txt* updates
        
        # if age_txt is starting this frame...
        if age_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            age_txt.frameNStart = frameN  # exact frame index
            age_txt.tStart = t  # local t and not account for scr refresh
            age_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(age_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'age_txt.started')
            # update status
            age_txt.status = STARTED
            age_txt.setAutoDraw(True)
        
        # if age_txt is active this frame...
        if age_txt.status == STARTED:
            # update params
            pass
        
        # *age_slider* updates
        
        # if age_slider is starting this frame...
        if age_slider.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            age_slider.frameNStart = frameN  # exact frame index
            age_slider.tStart = t  # local t and not account for scr refresh
            age_slider.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(age_slider, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'age_slider.started')
            # update status
            age_slider.status = STARTED
            age_slider.setAutoDraw(True)
        
        # if age_slider is active this frame...
        if age_slider.status == STARTED:
            # update params
            pass
        
        # *continue_txt_4* updates
        
        # if continue_txt_4 is starting this frame...
        if continue_txt_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_4.frameNStart = frameN  # exact frame index
            continue_txt_4.tStart = t  # local t and not account for scr refresh
            continue_txt_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_4.started')
            # update status
            continue_txt_4.status = STARTED
            continue_txt_4.setAutoDraw(True)
        
        # if continue_txt_4 is active this frame...
        if continue_txt_4.status == STARTED:
            # update params
            pass
        # *mouse_4* updates
        
        # if mouse_4 is starting this frame...
        if mouse_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_4.frameNStart = frameN  # exact frame index
            mouse_4.tStart = t  # local t and not account for scr refresh
            mouse_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_4, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_4.status = STARTED
            mouse_4.mouseClock.reset()
            prevButtonState = mouse_4.getPressed()  # if button is down already this ISN'T a new click
        if mouse_4.status == STARTED:  # only update if started and not finished!
            buttons = mouse_4.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_4, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_4):
                            gotValidClick = True
                            mouse_4.clicked_name.append(obj.name)
                    x, y = mouse_4.getPos()
                    mouse_4.x.append(x)
                    mouse_4.y.append(y)
                    buttons = mouse_4.getPressed()
                    mouse_4.leftButton.append(buttons[0])
                    mouse_4.midButton.append(buttons[1])
                    mouse_4.rightButton.append(buttons[2])
                    mouse_4.time.append(mouse_4.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ageComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "age" ---
    for thisComponent in ageComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('age.stopped', globalClock.getTime(format='float'))
    thisExp.addData('age_slider.response', age_slider.getRating())
    thisExp.addData('age_slider.rt', age_slider.getRT())
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_4.x', mouse_4.x)
    thisExp.addData('mouse_4.y', mouse_4.y)
    thisExp.addData('mouse_4.leftButton', mouse_4.leftButton)
    thisExp.addData('mouse_4.midButton', mouse_4.midButton)
    thisExp.addData('mouse_4.rightButton', mouse_4.rightButton)
    thisExp.addData('mouse_4.time', mouse_4.time)
    thisExp.addData('mouse_4.clicked_name', mouse_4.clicked_name)
    thisExp.nextEntry()
    # the Routine "age" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "gender" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('gender.started', globalClock.getTime(format='float'))
    # setup some python lists for storing info about the mouse_5
    mouse_5.x = []
    mouse_5.y = []
    mouse_5.leftButton = []
    mouse_5.midButton = []
    mouse_5.rightButton = []
    mouse_5.time = []
    mouse_5.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    genderComponents = [gender_txt, woman, man, nonbinary, other, mouse_5]
    for thisComponent in genderComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "gender" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *gender_txt* updates
        
        # if gender_txt is starting this frame...
        if gender_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            gender_txt.frameNStart = frameN  # exact frame index
            gender_txt.tStart = t  # local t and not account for scr refresh
            gender_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(gender_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'gender_txt.started')
            # update status
            gender_txt.status = STARTED
            gender_txt.setAutoDraw(True)
        
        # if gender_txt is active this frame...
        if gender_txt.status == STARTED:
            # update params
            pass
        
        # *woman* updates
        
        # if woman is starting this frame...
        if woman.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            woman.frameNStart = frameN  # exact frame index
            woman.tStart = t  # local t and not account for scr refresh
            woman.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(woman, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'woman.started')
            # update status
            woman.status = STARTED
            woman.setAutoDraw(True)
        
        # if woman is active this frame...
        if woman.status == STARTED:
            # update params
            pass
        
        # *man* updates
        
        # if man is starting this frame...
        if man.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            man.frameNStart = frameN  # exact frame index
            man.tStart = t  # local t and not account for scr refresh
            man.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(man, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'man.started')
            # update status
            man.status = STARTED
            man.setAutoDraw(True)
        
        # if man is active this frame...
        if man.status == STARTED:
            # update params
            pass
        
        # *nonbinary* updates
        
        # if nonbinary is starting this frame...
        if nonbinary.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            nonbinary.frameNStart = frameN  # exact frame index
            nonbinary.tStart = t  # local t and not account for scr refresh
            nonbinary.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(nonbinary, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'nonbinary.started')
            # update status
            nonbinary.status = STARTED
            nonbinary.setAutoDraw(True)
        
        # if nonbinary is active this frame...
        if nonbinary.status == STARTED:
            # update params
            pass
        
        # *other* updates
        
        # if other is starting this frame...
        if other.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            other.frameNStart = frameN  # exact frame index
            other.tStart = t  # local t and not account for scr refresh
            other.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(other, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'other.started')
            # update status
            other.status = STARTED
            other.setAutoDraw(True)
        
        # if other is active this frame...
        if other.status == STARTED:
            # update params
            pass
        # *mouse_5* updates
        
        # if mouse_5 is starting this frame...
        if mouse_5.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_5.frameNStart = frameN  # exact frame index
            mouse_5.tStart = t  # local t and not account for scr refresh
            mouse_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_5, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_5.status = STARTED
            mouse_5.mouseClock.reset()
            prevButtonState = mouse_5.getPressed()  # if button is down already this ISN'T a new click
        if mouse_5.status == STARTED:  # only update if started and not finished!
            buttons = mouse_5.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames([woman, man, nonbinary, other], namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_5):
                            gotValidClick = True
                            mouse_5.clicked_name.append(obj.name)
                    x, y = mouse_5.getPos()
                    mouse_5.x.append(x)
                    mouse_5.y.append(y)
                    buttons = mouse_5.getPressed()
                    mouse_5.leftButton.append(buttons[0])
                    mouse_5.midButton.append(buttons[1])
                    mouse_5.rightButton.append(buttons[2])
                    mouse_5.time.append(mouse_5.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in genderComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "gender" ---
    for thisComponent in genderComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('gender.stopped', globalClock.getTime(format='float'))
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_5.x', mouse_5.x)
    thisExp.addData('mouse_5.y', mouse_5.y)
    thisExp.addData('mouse_5.leftButton', mouse_5.leftButton)
    thisExp.addData('mouse_5.midButton', mouse_5.midButton)
    thisExp.addData('mouse_5.rightButton', mouse_5.rightButton)
    thisExp.addData('mouse_5.time', mouse_5.time)
    thisExp.addData('mouse_5.clicked_name', mouse_5.clicked_name)
    thisExp.nextEntry()
    # the Routine "gender" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "handedness" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('handedness.started', globalClock.getTime(format='float'))
    # setup some python lists for storing info about the mouse_6
    mouse_6.x = []
    mouse_6.y = []
    mouse_6.leftButton = []
    mouse_6.midButton = []
    mouse_6.rightButton = []
    mouse_6.time = []
    mouse_6.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    handednessComponents = [handedness_txt, right_hand, left_hand, ambidx, mouse_6]
    for thisComponent in handednessComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "handedness" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *handedness_txt* updates
        
        # if handedness_txt is starting this frame...
        if handedness_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            handedness_txt.frameNStart = frameN  # exact frame index
            handedness_txt.tStart = t  # local t and not account for scr refresh
            handedness_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(handedness_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'handedness_txt.started')
            # update status
            handedness_txt.status = STARTED
            handedness_txt.setAutoDraw(True)
        
        # if handedness_txt is active this frame...
        if handedness_txt.status == STARTED:
            # update params
            pass
        
        # *right_hand* updates
        
        # if right_hand is starting this frame...
        if right_hand.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            right_hand.frameNStart = frameN  # exact frame index
            right_hand.tStart = t  # local t and not account for scr refresh
            right_hand.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(right_hand, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'right_hand.started')
            # update status
            right_hand.status = STARTED
            right_hand.setAutoDraw(True)
        
        # if right_hand is active this frame...
        if right_hand.status == STARTED:
            # update params
            pass
        
        # *left_hand* updates
        
        # if left_hand is starting this frame...
        if left_hand.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            left_hand.frameNStart = frameN  # exact frame index
            left_hand.tStart = t  # local t and not account for scr refresh
            left_hand.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(left_hand, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'left_hand.started')
            # update status
            left_hand.status = STARTED
            left_hand.setAutoDraw(True)
        
        # if left_hand is active this frame...
        if left_hand.status == STARTED:
            # update params
            pass
        
        # *ambidx* updates
        
        # if ambidx is starting this frame...
        if ambidx.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            ambidx.frameNStart = frameN  # exact frame index
            ambidx.tStart = t  # local t and not account for scr refresh
            ambidx.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ambidx, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'ambidx.started')
            # update status
            ambidx.status = STARTED
            ambidx.setAutoDraw(True)
        
        # if ambidx is active this frame...
        if ambidx.status == STARTED:
            # update params
            pass
        # *mouse_6* updates
        
        # if mouse_6 is starting this frame...
        if mouse_6.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_6.frameNStart = frameN  # exact frame index
            mouse_6.tStart = t  # local t and not account for scr refresh
            mouse_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_6, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_6.status = STARTED
            mouse_6.mouseClock.reset()
            prevButtonState = mouse_6.getPressed()  # if button is down already this ISN'T a new click
        if mouse_6.status == STARTED:  # only update if started and not finished!
            buttons = mouse_6.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames([right_hand, left_hand, ambidx], namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_6):
                            gotValidClick = True
                            mouse_6.clicked_name.append(obj.name)
                    x, y = mouse_6.getPos()
                    mouse_6.x.append(x)
                    mouse_6.y.append(y)
                    buttons = mouse_6.getPressed()
                    mouse_6.leftButton.append(buttons[0])
                    mouse_6.midButton.append(buttons[1])
                    mouse_6.rightButton.append(buttons[2])
                    mouse_6.time.append(mouse_6.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in handednessComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "handedness" ---
    for thisComponent in handednessComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('handedness.stopped', globalClock.getTime(format='float'))
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_6.x', mouse_6.x)
    thisExp.addData('mouse_6.y', mouse_6.y)
    thisExp.addData('mouse_6.leftButton', mouse_6.leftButton)
    thisExp.addData('mouse_6.midButton', mouse_6.midButton)
    thisExp.addData('mouse_6.rightButton', mouse_6.rightButton)
    thisExp.addData('mouse_6.time', mouse_6.time)
    thisExp.addData('mouse_6.clicked_name', mouse_6.clicked_name)
    thisExp.nextEntry()
    # the Routine "handedness" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "short_instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('short_instructions.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from short_instr
    if instructions == 1:
        continueRoutine = False
    # setup some python lists for storing info about the mouse_25
    mouse_25.x = []
    mouse_25.y = []
    mouse_25.leftButton = []
    mouse_25.midButton = []
    mouse_25.rightButton = []
    mouse_25.time = []
    mouse_25.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    short_instructionsComponents = [short_instructions_txt, continue_txt_21, mouse_25]
    for thisComponent in short_instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "short_instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *short_instructions_txt* updates
        
        # if short_instructions_txt is starting this frame...
        if short_instructions_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            short_instructions_txt.frameNStart = frameN  # exact frame index
            short_instructions_txt.tStart = t  # local t and not account for scr refresh
            short_instructions_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(short_instructions_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'short_instructions_txt.started')
            # update status
            short_instructions_txt.status = STARTED
            short_instructions_txt.setAutoDraw(True)
        
        # if short_instructions_txt is active this frame...
        if short_instructions_txt.status == STARTED:
            # update params
            pass
        
        # *continue_txt_21* updates
        
        # if continue_txt_21 is starting this frame...
        if continue_txt_21.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_21.frameNStart = frameN  # exact frame index
            continue_txt_21.tStart = t  # local t and not account for scr refresh
            continue_txt_21.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_21, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_21.started')
            # update status
            continue_txt_21.status = STARTED
            continue_txt_21.setAutoDraw(True)
        
        # if continue_txt_21 is active this frame...
        if continue_txt_21.status == STARTED:
            # update params
            pass
        # *mouse_25* updates
        
        # if mouse_25 is starting this frame...
        if mouse_25.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_25.frameNStart = frameN  # exact frame index
            mouse_25.tStart = t  # local t and not account for scr refresh
            mouse_25.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_25, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_25.status = STARTED
            mouse_25.mouseClock.reset()
            prevButtonState = mouse_25.getPressed()  # if button is down already this ISN'T a new click
        if mouse_25.status == STARTED:  # only update if started and not finished!
            buttons = mouse_25.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_21, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_25):
                            gotValidClick = True
                            mouse_25.clicked_name.append(obj.name)
                    x, y = mouse_25.getPos()
                    mouse_25.x.append(x)
                    mouse_25.y.append(y)
                    buttons = mouse_25.getPressed()
                    mouse_25.leftButton.append(buttons[0])
                    mouse_25.midButton.append(buttons[1])
                    mouse_25.rightButton.append(buttons[2])
                    mouse_25.time.append(mouse_25.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in short_instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "short_instructions" ---
    for thisComponent in short_instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('short_instructions.stopped', globalClock.getTime(format='float'))
    # Run 'End Routine' code from start_session_timer_2
    thisExp.addData('session_start', data.getDateStr())
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_25.x', mouse_25.x)
    thisExp.addData('mouse_25.y', mouse_25.y)
    thisExp.addData('mouse_25.leftButton', mouse_25.leftButton)
    thisExp.addData('mouse_25.midButton', mouse_25.midButton)
    thisExp.addData('mouse_25.rightButton', mouse_25.rightButton)
    thisExp.addData('mouse_25.time', mouse_25.time)
    thisExp.addData('mouse_25.clicked_name', mouse_25.clicked_name)
    thisExp.nextEntry()
    # the Routine "short_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "pre_instr" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('pre_instr.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from long_instr
    if instructions == 0:
        continueRoutine = False
    # setup some python lists for storing info about the mouse_22
    mouse_22.x = []
    mouse_22.y = []
    mouse_22.leftButton = []
    mouse_22.midButton = []
    mouse_22.rightButton = []
    mouse_22.time = []
    mouse_22.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    pre_instrComponents = [pre_instr_txt, continue_txt_18, mouse_22]
    for thisComponent in pre_instrComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pre_instr" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *pre_instr_txt* updates
        
        # if pre_instr_txt is starting this frame...
        if pre_instr_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            pre_instr_txt.frameNStart = frameN  # exact frame index
            pre_instr_txt.tStart = t  # local t and not account for scr refresh
            pre_instr_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pre_instr_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'pre_instr_txt.started')
            # update status
            pre_instr_txt.status = STARTED
            pre_instr_txt.setAutoDraw(True)
        
        # if pre_instr_txt is active this frame...
        if pre_instr_txt.status == STARTED:
            # update params
            pass
        
        # *continue_txt_18* updates
        
        # if continue_txt_18 is starting this frame...
        if continue_txt_18.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_18.frameNStart = frameN  # exact frame index
            continue_txt_18.tStart = t  # local t and not account for scr refresh
            continue_txt_18.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_18, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_18.started')
            # update status
            continue_txt_18.status = STARTED
            continue_txt_18.setAutoDraw(True)
        
        # if continue_txt_18 is active this frame...
        if continue_txt_18.status == STARTED:
            # update params
            pass
        # *mouse_22* updates
        
        # if mouse_22 is starting this frame...
        if mouse_22.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_22.frameNStart = frameN  # exact frame index
            mouse_22.tStart = t  # local t and not account for scr refresh
            mouse_22.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_22, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_22.status = STARTED
            mouse_22.mouseClock.reset()
            prevButtonState = mouse_22.getPressed()  # if button is down already this ISN'T a new click
        if mouse_22.status == STARTED:  # only update if started and not finished!
            buttons = mouse_22.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_18, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_22):
                            gotValidClick = True
                            mouse_22.clicked_name.append(obj.name)
                    x, y = mouse_22.getPos()
                    mouse_22.x.append(x)
                    mouse_22.y.append(y)
                    buttons = mouse_22.getPressed()
                    mouse_22.leftButton.append(buttons[0])
                    mouse_22.midButton.append(buttons[1])
                    mouse_22.rightButton.append(buttons[2])
                    mouse_22.time.append(mouse_22.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pre_instrComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pre_instr" ---
    for thisComponent in pre_instrComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('pre_instr.stopped', globalClock.getTime(format='float'))
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_22.x', mouse_22.x)
    thisExp.addData('mouse_22.y', mouse_22.y)
    thisExp.addData('mouse_22.leftButton', mouse_22.leftButton)
    thisExp.addData('mouse_22.midButton', mouse_22.midButton)
    thisExp.addData('mouse_22.rightButton', mouse_22.rightButton)
    thisExp.addData('mouse_22.time', mouse_22.time)
    thisExp.addData('mouse_22.clicked_name', mouse_22.clicked_name)
    thisExp.nextEntry()
    # the Routine "pre_instr" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions_1.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from long_instr_2
    if instructions == 0:
        continueRoutine = False
    # setup some python lists for storing info about the mouse_2
    mouse_2.x = []
    mouse_2.y = []
    mouse_2.leftButton = []
    mouse_2.midButton = []
    mouse_2.rightButton = []
    mouse_2.time = []
    mouse_2.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    instructions_1Components = [instruction_txt, continue_txt_2, mouse_2]
    for thisComponent in instructions_1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruction_txt* updates
        
        # if instruction_txt is starting this frame...
        if instruction_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_txt.frameNStart = frameN  # exact frame index
            instruction_txt.tStart = t  # local t and not account for scr refresh
            instruction_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_txt.started')
            # update status
            instruction_txt.status = STARTED
            instruction_txt.setAutoDraw(True)
        
        # if instruction_txt is active this frame...
        if instruction_txt.status == STARTED:
            # update params
            pass
        
        # *continue_txt_2* updates
        
        # if continue_txt_2 is starting this frame...
        if continue_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_2.frameNStart = frameN  # exact frame index
            continue_txt_2.tStart = t  # local t and not account for scr refresh
            continue_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_2.started')
            # update status
            continue_txt_2.status = STARTED
            continue_txt_2.setAutoDraw(True)
        
        # if continue_txt_2 is active this frame...
        if continue_txt_2.status == STARTED:
            # update params
            pass
        # *mouse_2* updates
        
        # if mouse_2 is starting this frame...
        if mouse_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_2.frameNStart = frameN  # exact frame index
            mouse_2.tStart = t  # local t and not account for scr refresh
            mouse_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_2.status = STARTED
            mouse_2.mouseClock.reset()
            prevButtonState = mouse_2.getPressed()  # if button is down already this ISN'T a new click
        if mouse_2.status == STARTED:  # only update if started and not finished!
            buttons = mouse_2.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_2, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_2):
                            gotValidClick = True
                            mouse_2.clicked_name.append(obj.name)
                    x, y = mouse_2.getPos()
                    mouse_2.x.append(x)
                    mouse_2.y.append(y)
                    buttons = mouse_2.getPressed()
                    mouse_2.leftButton.append(buttons[0])
                    mouse_2.midButton.append(buttons[1])
                    mouse_2.rightButton.append(buttons[2])
                    mouse_2.time.append(mouse_2.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_1" ---
    for thisComponent in instructions_1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions_1.stopped', globalClock.getTime(format='float'))
    # Run 'End Routine' code from save_winsize
    thisExp.addData('session_start', core.getTime())
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_2.x', mouse_2.x)
    thisExp.addData('mouse_2.y', mouse_2.y)
    thisExp.addData('mouse_2.leftButton', mouse_2.leftButton)
    thisExp.addData('mouse_2.midButton', mouse_2.midButton)
    thisExp.addData('mouse_2.rightButton', mouse_2.rightButton)
    thisExp.addData('mouse_2.time', mouse_2.time)
    thisExp.addData('mouse_2.clicked_name', mouse_2.clicked_name)
    thisExp.nextEntry()
    # the Routine "instructions_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions_2.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from long_instr_3
    if instructions == 0:
        continueRoutine = False
    # setup some python lists for storing info about the mouse_11
    mouse_11.x = []
    mouse_11.y = []
    mouse_11.leftButton = []
    mouse_11.midButton = []
    mouse_11.rightButton = []
    mouse_11.time = []
    mouse_11.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    instructions_2Components = [example_r, instruction_txt_2, continue_txt_9, mouse_11]
    for thisComponent in instructions_2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *example_r* updates
        
        # if example_r is starting this frame...
        if example_r.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            example_r.frameNStart = frameN  # exact frame index
            example_r.tStart = t  # local t and not account for scr refresh
            example_r.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(example_r, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'example_r.started')
            # update status
            example_r.status = STARTED
            example_r.setAutoDraw(True)
        
        # if example_r is active this frame...
        if example_r.status == STARTED:
            # update params
            pass
        
        # *instruction_txt_2* updates
        
        # if instruction_txt_2 is starting this frame...
        if instruction_txt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_txt_2.frameNStart = frameN  # exact frame index
            instruction_txt_2.tStart = t  # local t and not account for scr refresh
            instruction_txt_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_txt_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_txt_2.started')
            # update status
            instruction_txt_2.status = STARTED
            instruction_txt_2.setAutoDraw(True)
        
        # if instruction_txt_2 is active this frame...
        if instruction_txt_2.status == STARTED:
            # update params
            pass
        
        # *continue_txt_9* updates
        
        # if continue_txt_9 is starting this frame...
        if continue_txt_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_9.frameNStart = frameN  # exact frame index
            continue_txt_9.tStart = t  # local t and not account for scr refresh
            continue_txt_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_9.started')
            # update status
            continue_txt_9.status = STARTED
            continue_txt_9.setAutoDraw(True)
        
        # if continue_txt_9 is active this frame...
        if continue_txt_9.status == STARTED:
            # update params
            pass
        # *mouse_11* updates
        
        # if mouse_11 is starting this frame...
        if mouse_11.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_11.frameNStart = frameN  # exact frame index
            mouse_11.tStart = t  # local t and not account for scr refresh
            mouse_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_11, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_11.status = STARTED
            mouse_11.mouseClock.reset()
            prevButtonState = mouse_11.getPressed()  # if button is down already this ISN'T a new click
        if mouse_11.status == STARTED:  # only update if started and not finished!
            buttons = mouse_11.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_9, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_11):
                            gotValidClick = True
                            mouse_11.clicked_name.append(obj.name)
                    x, y = mouse_11.getPos()
                    mouse_11.x.append(x)
                    mouse_11.y.append(y)
                    buttons = mouse_11.getPressed()
                    mouse_11.leftButton.append(buttons[0])
                    mouse_11.midButton.append(buttons[1])
                    mouse_11.rightButton.append(buttons[2])
                    mouse_11.time.append(mouse_11.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_2" ---
    for thisComponent in instructions_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions_2.stopped', globalClock.getTime(format='float'))
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_11.x', mouse_11.x)
    thisExp.addData('mouse_11.y', mouse_11.y)
    thisExp.addData('mouse_11.leftButton', mouse_11.leftButton)
    thisExp.addData('mouse_11.midButton', mouse_11.midButton)
    thisExp.addData('mouse_11.rightButton', mouse_11.rightButton)
    thisExp.addData('mouse_11.time', mouse_11.time)
    thisExp.addData('mouse_11.clicked_name', mouse_11.clicked_name)
    thisExp.nextEntry()
    # the Routine "instructions_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_3" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions_3.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from long_instr_4
    if instructions == 0:
        continueRoutine = False
    # setup some python lists for storing info about the mouse_12
    mouse_12.x = []
    mouse_12.y = []
    mouse_12.leftButton = []
    mouse_12.midButton = []
    mouse_12.rightButton = []
    mouse_12.time = []
    mouse_12.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    instructions_3Components = [example_l, instruction_txt_3, continue_txt_10, mouse_12]
    for thisComponent in instructions_3Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_3" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *example_l* updates
        
        # if example_l is starting this frame...
        if example_l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            example_l.frameNStart = frameN  # exact frame index
            example_l.tStart = t  # local t and not account for scr refresh
            example_l.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(example_l, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'example_l.started')
            # update status
            example_l.status = STARTED
            example_l.setAutoDraw(True)
        
        # if example_l is active this frame...
        if example_l.status == STARTED:
            # update params
            pass
        
        # *instruction_txt_3* updates
        
        # if instruction_txt_3 is starting this frame...
        if instruction_txt_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_txt_3.frameNStart = frameN  # exact frame index
            instruction_txt_3.tStart = t  # local t and not account for scr refresh
            instruction_txt_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_txt_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_txt_3.started')
            # update status
            instruction_txt_3.status = STARTED
            instruction_txt_3.setAutoDraw(True)
        
        # if instruction_txt_3 is active this frame...
        if instruction_txt_3.status == STARTED:
            # update params
            pass
        
        # *continue_txt_10* updates
        
        # if continue_txt_10 is starting this frame...
        if continue_txt_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_10.frameNStart = frameN  # exact frame index
            continue_txt_10.tStart = t  # local t and not account for scr refresh
            continue_txt_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_10, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_10.started')
            # update status
            continue_txt_10.status = STARTED
            continue_txt_10.setAutoDraw(True)
        
        # if continue_txt_10 is active this frame...
        if continue_txt_10.status == STARTED:
            # update params
            pass
        # *mouse_12* updates
        
        # if mouse_12 is starting this frame...
        if mouse_12.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_12.frameNStart = frameN  # exact frame index
            mouse_12.tStart = t  # local t and not account for scr refresh
            mouse_12.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_12, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_12.status = STARTED
            mouse_12.mouseClock.reset()
            prevButtonState = mouse_12.getPressed()  # if button is down already this ISN'T a new click
        if mouse_12.status == STARTED:  # only update if started and not finished!
            buttons = mouse_12.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_10, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_12):
                            gotValidClick = True
                            mouse_12.clicked_name.append(obj.name)
                    x, y = mouse_12.getPos()
                    mouse_12.x.append(x)
                    mouse_12.y.append(y)
                    buttons = mouse_12.getPressed()
                    mouse_12.leftButton.append(buttons[0])
                    mouse_12.midButton.append(buttons[1])
                    mouse_12.rightButton.append(buttons[2])
                    mouse_12.time.append(mouse_12.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_3" ---
    for thisComponent in instructions_3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions_3.stopped', globalClock.getTime(format='float'))
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_12.x', mouse_12.x)
    thisExp.addData('mouse_12.y', mouse_12.y)
    thisExp.addData('mouse_12.leftButton', mouse_12.leftButton)
    thisExp.addData('mouse_12.midButton', mouse_12.midButton)
    thisExp.addData('mouse_12.rightButton', mouse_12.rightButton)
    thisExp.addData('mouse_12.time', mouse_12.time)
    thisExp.addData('mouse_12.clicked_name', mouse_12.clicked_name)
    thisExp.nextEntry()
    # the Routine "instructions_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_4" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions_4.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from long_instr_5
    if instructions == 0:
        continueRoutine = False
    # setup some python lists for storing info about the mouse_13
    mouse_13.x = []
    mouse_13.y = []
    mouse_13.leftButton = []
    mouse_13.midButton = []
    mouse_13.rightButton = []
    mouse_13.time = []
    mouse_13.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    instructions_4Components = [instruction_txt_4, continue_txt_11, mouse_13]
    for thisComponent in instructions_4Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_4" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruction_txt_4* updates
        
        # if instruction_txt_4 is starting this frame...
        if instruction_txt_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_txt_4.frameNStart = frameN  # exact frame index
            instruction_txt_4.tStart = t  # local t and not account for scr refresh
            instruction_txt_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_txt_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_txt_4.started')
            # update status
            instruction_txt_4.status = STARTED
            instruction_txt_4.setAutoDraw(True)
        
        # if instruction_txt_4 is active this frame...
        if instruction_txt_4.status == STARTED:
            # update params
            pass
        
        # *continue_txt_11* updates
        
        # if continue_txt_11 is starting this frame...
        if continue_txt_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_11.frameNStart = frameN  # exact frame index
            continue_txt_11.tStart = t  # local t and not account for scr refresh
            continue_txt_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_11, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_11.started')
            # update status
            continue_txt_11.status = STARTED
            continue_txt_11.setAutoDraw(True)
        
        # if continue_txt_11 is active this frame...
        if continue_txt_11.status == STARTED:
            # update params
            pass
        # *mouse_13* updates
        
        # if mouse_13 is starting this frame...
        if mouse_13.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_13.frameNStart = frameN  # exact frame index
            mouse_13.tStart = t  # local t and not account for scr refresh
            mouse_13.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_13, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_13.status = STARTED
            mouse_13.mouseClock.reset()
            prevButtonState = mouse_13.getPressed()  # if button is down already this ISN'T a new click
        if mouse_13.status == STARTED:  # only update if started and not finished!
            buttons = mouse_13.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_11, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_13):
                            gotValidClick = True
                            mouse_13.clicked_name.append(obj.name)
                    x, y = mouse_13.getPos()
                    mouse_13.x.append(x)
                    mouse_13.y.append(y)
                    buttons = mouse_13.getPressed()
                    mouse_13.leftButton.append(buttons[0])
                    mouse_13.midButton.append(buttons[1])
                    mouse_13.rightButton.append(buttons[2])
                    mouse_13.time.append(mouse_13.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_4Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_4" ---
    for thisComponent in instructions_4Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions_4.stopped', globalClock.getTime(format='float'))
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_13.x', mouse_13.x)
    thisExp.addData('mouse_13.y', mouse_13.y)
    thisExp.addData('mouse_13.leftButton', mouse_13.leftButton)
    thisExp.addData('mouse_13.midButton', mouse_13.midButton)
    thisExp.addData('mouse_13.rightButton', mouse_13.rightButton)
    thisExp.addData('mouse_13.time', mouse_13.time)
    thisExp.addData('mouse_13.clicked_name', mouse_13.clicked_name)
    thisExp.nextEntry()
    # the Routine "instructions_4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice = data.TrialHandler(nReps=instructions, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('pregen_sequence_' + str(int(expInfo['participant'][-1])) + '.xlsx', selection='36:41'),
        seed=None, name='practice')
    thisExp.addLoop(practice)  # add the loop to the experiment
    thisPractice = practice.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
    if thisPractice != None:
        for paramName in thisPractice:
            globals()[paramName] = thisPractice[paramName]
    
    for thisPractice in practice:
        currentLoop = practice
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
        if thisPractice != None:
            for paramName in thisPractice:
                globals()[paramName] = thisPractice[paramName]
        
        # --- Prepare to start Routine "fix_practice" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fix_practice.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from long_instr_6
        if instructions == 0:
            continueRoutine = False
        # Run 'Begin Routine' code from mouse_visible_5
        win.mouseVisible = False
        mouse.setPos(newPos=(0, 0))
        # Run 'Begin Routine' code from set_contrast_side_2
        if eccentricity == -15:
            leftCont = baseContrast+contrastDelta
            rightCont = baseContrast
        elif eccentricity == 15:
            leftCont = baseContrast
            rightCont = baseContrast+contrastDelta
        
        signed_contrast=rightCont-leftCont
        
        
        # Save all these variables to the log
        thisExp.addData("signed_contrast", signed_contrast)
        thisExp.addData("leftCont", leftCont)
        thisExp.addData("rightCont", rightCont)
        fixation_5.setColor([1,1,1], colorSpace='rgb')
        fixation_5.setSize((0.75, 0.75))
        # keep track of which components have finished
        fix_practiceComponents = [fixation_5]
        for thisComponent in fix_practiceComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fix_practice" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from mouse_visible_5
            mouse.setPos(newPos=(0, 0))
            
            # *fixation_5* updates
            
            # if fixation_5 is starting this frame...
            if fixation_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_5.frameNStart = frameN  # exact frame index
                fixation_5.tStart = t  # local t and not account for scr refresh
                fixation_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_5.started')
                # update status
                fixation_5.status = STARTED
                fixation_5.setAutoDraw(True)
            
            # if fixation_5 is active this frame...
            if fixation_5.status == STARTED:
                # update params
                pass
            
            # if fixation_5 is stopping this frame...
            if fixation_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_5.tStartRefresh + q-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_5.tStop = t  # not accounting for scr refresh
                    fixation_5.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_5.stopped')
                    # update status
                    fixation_5.status = FINISHED
                    fixation_5.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fix_practiceComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fix_practice" ---
        for thisComponent in fix_practiceComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fix_practice.stopped', globalClock.getTime(format='float'))
        # the Routine "fix_practice" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial_practice" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial_practice.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from long_instr_7
        if instructions == 0:
            continueRoutine = False
        # Run 'Begin Routine' code from dragging_code_3
        mouse_21.setPos(newPos=(0, 0))
        
        r_lim_corr = 0.05
        l_lim_corr = -0.05
        r_lim_wrong = 1236
        l_lim_wrong = -1236
        
        mouserec = mouse_21.getPos()
        moved = False
        
        dot_trajectory = []
        dot_3.setPos((618, 0))
        fixation_6.setColor([0,0,0], colorSpace='rgb')
        fixation_6.setSize((0.75, 0.75))
        grating_l_3.setContrast(leftCont)
        grating_l_3.setPos((-618, 0))
        grating_l_3.setPhase(random.random()*360)
        grating_r_3.setContrast(rightCont)
        grating_r_3.setPos((618, 0))
        grating_r_3.setPhase(random.random()*360)
        # setup some python lists for storing info about the mouse_21
        mouse_21.x = []
        mouse_21.y = []
        mouse_21.leftButton = []
        mouse_21.midButton = []
        mouse_21.rightButton = []
        mouse_21.time = []
        gotValidClick = False  # until a click is received
        sound_trial_start_3.setSound('5000', secs=0.1, hamming=True)
        sound_trial_start_3.setVolume(0.1, log=False)
        sound_trial_start_3.seek(0)
        sound_no_resp_3.setSound('567', secs=0.5, hamming=True)
        sound_no_resp_3.setVolume(0.1, log=False)
        sound_no_resp_3.seek(0)
        # keep track of which components have finished
        trial_practiceComponents = [dot_3, fixation_6, grating_l_3, grating_r_3, mouse_21, sound_trial_start_3, sound_no_resp_3]
        for thisComponent in trial_practiceComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_practice" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from dragging_code_3
            x = mouse_21.getPos()[0]
            #y = mouse.getPos()[1]
            
            if moved == False:
                 mouseloc = mouse_21.getPos()
                 if mouseloc[0] != mouserec[0] or mouseloc[1] != mouserec[1]:
                      moved = True
                      thisExp.addData('reaction_time',round(t*1000))
            
            x_r = x + 618
            x_l = x - 618
            
            grating_r_3.pos = (x_r,0)
            grating_l_3.pos = (x_l,0)
            
            if grating_r_3.overlaps(dot_3):
                dot_3.pos = grating_r_3.pos
            
            elif grating_l_3.overlaps(dot_3):
                dot_3.pos = grating_l_3.pos
            
            dot_trajectory.append(dot_3.pos[0])
            
            if eccentricity > 0:
                if grating_r_3.pos[0] <= r_lim_corr:
                    correct = 1
                    timeout=0
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
                elif grating_r_3.pos[0] >= r_lim_wrong:
                    correct = 0
                    timeout=0
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
            elif eccentricity < 0:
                if grating_l_3.pos[0] >= l_lim_corr:
                    correct = 1
                    timeout=0
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
                elif grating_l_3.pos[0] <= l_lim_wrong:
                    correct = 0
                    timeout=0
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
            
            
            if sound_no_resp_3.status == STARTED:
                correct = 'NaN'
                timeout = 1
            
            
            
            # *dot_3* updates
            
            # if dot_3 is starting this frame...
            if dot_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot_3.frameNStart = frameN  # exact frame index
                dot_3.tStart = t  # local t and not account for scr refresh
                dot_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dot_3.started')
                # update status
                dot_3.status = STARTED
                dot_3.setAutoDraw(True)
            
            # if dot_3 is active this frame...
            if dot_3.status == STARTED:
                # update params
                pass
            
            # if dot_3 is stopping this frame...
            if dot_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dot_3.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    dot_3.tStop = t  # not accounting for scr refresh
                    dot_3.tStopRefresh = tThisFlipGlobal  # on global time
                    dot_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dot_3.stopped')
                    # update status
                    dot_3.status = FINISHED
                    dot_3.setAutoDraw(False)
            
            # *fixation_6* updates
            
            # if fixation_6 is starting this frame...
            if fixation_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_6.frameNStart = frameN  # exact frame index
                fixation_6.tStart = t  # local t and not account for scr refresh
                fixation_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_6.started')
                # update status
                fixation_6.status = STARTED
                fixation_6.setAutoDraw(True)
            
            # if fixation_6 is active this frame...
            if fixation_6.status == STARTED:
                # update params
                pass
            
            # if fixation_6 is stopping this frame...
            if fixation_6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_6.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_6.tStop = t  # not accounting for scr refresh
                    fixation_6.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_6.stopped')
                    # update status
                    fixation_6.status = FINISHED
                    fixation_6.setAutoDraw(False)
            
            # *grating_l_3* updates
            
            # if grating_l_3 is starting this frame...
            if grating_l_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                grating_l_3.frameNStart = frameN  # exact frame index
                grating_l_3.tStart = t  # local t and not account for scr refresh
                grating_l_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grating_l_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'grating_l_3.started')
                # update status
                grating_l_3.status = STARTED
                grating_l_3.setAutoDraw(True)
            
            # if grating_l_3 is active this frame...
            if grating_l_3.status == STARTED:
                # update params
                pass
            
            # if grating_l_3 is stopping this frame...
            if grating_l_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grating_l_3.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    grating_l_3.tStop = t  # not accounting for scr refresh
                    grating_l_3.tStopRefresh = tThisFlipGlobal  # on global time
                    grating_l_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grating_l_3.stopped')
                    # update status
                    grating_l_3.status = FINISHED
                    grating_l_3.setAutoDraw(False)
            
            # *grating_r_3* updates
            
            # if grating_r_3 is starting this frame...
            if grating_r_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                grating_r_3.frameNStart = frameN  # exact frame index
                grating_r_3.tStart = t  # local t and not account for scr refresh
                grating_r_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grating_r_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'grating_r_3.started')
                # update status
                grating_r_3.status = STARTED
                grating_r_3.setAutoDraw(True)
            
            # if grating_r_3 is active this frame...
            if grating_r_3.status == STARTED:
                # update params
                pass
            
            # if grating_r_3 is stopping this frame...
            if grating_r_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grating_r_3.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    grating_r_3.tStop = t  # not accounting for scr refresh
                    grating_r_3.tStopRefresh = tThisFlipGlobal  # on global time
                    grating_r_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grating_r_3.stopped')
                    # update status
                    grating_r_3.status = FINISHED
                    grating_r_3.setAutoDraw(False)
            # *mouse_21* updates
            
            # if mouse_21 is starting this frame...
            if mouse_21.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_21.frameNStart = frameN  # exact frame index
                mouse_21.tStart = t  # local t and not account for scr refresh
                mouse_21.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_21, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_21.started', t)
                # update status
                mouse_21.status = STARTED
                prevButtonState = mouse_21.getPressed()  # if button is down already this ISN'T a new click
            
            # if mouse_21 is stopping this frame...
            if mouse_21.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > mouse_21.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    mouse_21.tStop = t  # not accounting for scr refresh
                    mouse_21.tStopRefresh = tThisFlipGlobal  # on global time
                    mouse_21.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('mouse_21.stopped', t)
                    # update status
                    mouse_21.status = FINISHED
            if mouse_21.status == STARTED:  # only update if started and not finished!
                x, y = mouse_21.getPos()
                mouse_21.x.append(x)
                mouse_21.y.append(y)
                buttons = mouse_21.getPressed()
                mouse_21.leftButton.append(buttons[0])
                mouse_21.midButton.append(buttons[1])
                mouse_21.rightButton.append(buttons[2])
                mouse_21.time.append(globalClock.getTime())
            
            # if sound_trial_start_3 is starting this frame...
            if sound_trial_start_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_trial_start_3.frameNStart = frameN  # exact frame index
                sound_trial_start_3.tStart = t  # local t and not account for scr refresh
                sound_trial_start_3.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_trial_start_3.started', tThisFlipGlobal)
                # update status
                sound_trial_start_3.status = STARTED
                sound_trial_start_3.play(when=win)  # sync with win flip
            
            # if sound_trial_start_3 is stopping this frame...
            if sound_trial_start_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_trial_start_3.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_trial_start_3.tStop = t  # not accounting for scr refresh
                    sound_trial_start_3.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_trial_start_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_trial_start_3.stopped')
                    # update status
                    sound_trial_start_3.status = FINISHED
                    sound_trial_start_3.stop()
            # update sound_trial_start_3 status according to whether it's playing
            if sound_trial_start_3.isPlaying:
                sound_trial_start_3.status = STARTED
            elif sound_trial_start_3.isFinished:
                sound_trial_start_3.status = FINISHED
            
            # if sound_no_resp_3 is starting this frame...
            if sound_no_resp_3.status == NOT_STARTED and tThisFlip >= 10-frameTolerance:
                # keep track of start time/frame for later
                sound_no_resp_3.frameNStart = frameN  # exact frame index
                sound_no_resp_3.tStart = t  # local t and not account for scr refresh
                sound_no_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_no_resp_3.started', tThisFlipGlobal)
                # update status
                sound_no_resp_3.status = STARTED
                sound_no_resp_3.play(when=win)  # sync with win flip
            
            # if sound_no_resp_3 is stopping this frame...
            if sound_no_resp_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_no_resp_3.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_no_resp_3.tStop = t  # not accounting for scr refresh
                    sound_no_resp_3.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_no_resp_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_no_resp_3.stopped')
                    # update status
                    sound_no_resp_3.status = FINISHED
                    sound_no_resp_3.stop()
            # update sound_no_resp_3 status according to whether it's playing
            if sound_no_resp_3.isPlaying:
                sound_no_resp_3.status = STARTED
            elif sound_no_resp_3.isFinished:
                sound_no_resp_3.status = FINISHED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_practiceComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_practice" ---
        for thisComponent in trial_practiceComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial_practice.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from dragging_code_3
        thisExp.addData("correct", correct)
        thisExp.addData('timeout', timeout)
        thisExp.addData("dot_trajectory", dot_trajectory)
        # store data for practice (TrialHandler)
        practice.addData('mouse_21.x', mouse_21.x)
        practice.addData('mouse_21.y', mouse_21.y)
        practice.addData('mouse_21.leftButton', mouse_21.leftButton)
        practice.addData('mouse_21.midButton', mouse_21.midButton)
        practice.addData('mouse_21.rightButton', mouse_21.rightButton)
        practice.addData('mouse_21.time', mouse_21.time)
        sound_trial_start_3.pause()  # ensure sound has stopped at end of Routine
        sound_no_resp_3.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.500000)
        
        # --- Prepare to start Routine "feedback_practice" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('feedback_practice.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from long_instr_8
        if instructions == 0:
            continueRoutine = False
        # Run 'Begin Routine' code from feedback_code_2
        mouse.setPos(newPos=(0, 0))
        
        if correct==1:
            fb_sound = "2000.wav"
            fb_sound_dur = 0.2
            fb_dur = 1
            fb_volume = 0.1
            
        elif correct==0:
            fb_sound = "whitenoise.wav"
            fb_sound_dur = 0.5
            fb_dur = 2
            fb_volume = 0.1
        
        else:
            fb_sound = 200
            fb_sound_dur = 0.1
            fb_dur = 1.5
            fb_volume = 0
        feedback_sound_2.setSound(fb_sound, secs=fb_sound_dur, hamming=True)
        feedback_sound_2.setVolume(fb_volume, log=False)
        feedback_sound_2.seek(0)
        # keep track of which components have finished
        feedback_practiceComponents = [feedback_sound_2, fixation_7]
        for thisComponent in feedback_practiceComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback_practice" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from feedback_code_2
            mouse.setPos(newPos=(0, 0))
            
            # if feedback_sound_2 is starting this frame...
            if feedback_sound_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_sound_2.frameNStart = frameN  # exact frame index
                feedback_sound_2.tStart = t  # local t and not account for scr refresh
                feedback_sound_2.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('feedback_sound_2.started', tThisFlipGlobal)
                # update status
                feedback_sound_2.status = STARTED
                feedback_sound_2.play(when=win)  # sync with win flip
            
            # if feedback_sound_2 is stopping this frame...
            if feedback_sound_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_sound_2.tStartRefresh + fb_sound_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_sound_2.tStop = t  # not accounting for scr refresh
                    feedback_sound_2.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_sound_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_sound_2.stopped')
                    # update status
                    feedback_sound_2.status = FINISHED
                    feedback_sound_2.stop()
            # update feedback_sound_2 status according to whether it's playing
            if feedback_sound_2.isPlaying:
                feedback_sound_2.status = STARTED
            elif feedback_sound_2.isFinished:
                feedback_sound_2.status = FINISHED
            
            # *fixation_7* updates
            
            # if fixation_7 is starting this frame...
            if fixation_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_7.frameNStart = frameN  # exact frame index
                fixation_7.tStart = t  # local t and not account for scr refresh
                fixation_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_7.started')
                # update status
                fixation_7.status = STARTED
                fixation_7.setAutoDraw(True)
            
            # if fixation_7 is active this frame...
            if fixation_7.status == STARTED:
                # update params
                pass
            
            # if fixation_7 is stopping this frame...
            if fixation_7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_7.tStartRefresh + fb_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_7.tStop = t  # not accounting for scr refresh
                    fixation_7.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_7.stopped')
                    # update status
                    fixation_7.status = FINISHED
                    fixation_7.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback_practiceComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback_practice" ---
        for thisComponent in feedback_practiceComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('feedback_practice.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from feedback_code_2
        #feedback_sound.sound
        feedback_sound_2.pause()  # ensure sound has stopped at end of Routine
        # the Routine "feedback_practice" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed instructions repeats of 'practice'
    
    
    # --- Prepare to start Routine "start_session" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('start_session.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from long_instr_9
    if instructions == 0:
        continueRoutine = False
    # Run 'Begin Routine' code from mouse_visible_2
    win.mouseVisible = True
    # setup some python lists for storing info about the mouse_14
    mouse_14.x = []
    mouse_14.y = []
    mouse_14.leftButton = []
    mouse_14.midButton = []
    mouse_14.rightButton = []
    mouse_14.time = []
    mouse_14.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    start_sessionComponents = [end_practice_txt, continue_txt_12, mouse_14]
    for thisComponent in start_sessionComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start_session" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_practice_txt* updates
        
        # if end_practice_txt is starting this frame...
        if end_practice_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_practice_txt.frameNStart = frameN  # exact frame index
            end_practice_txt.tStart = t  # local t and not account for scr refresh
            end_practice_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_practice_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_practice_txt.started')
            # update status
            end_practice_txt.status = STARTED
            end_practice_txt.setAutoDraw(True)
        
        # if end_practice_txt is active this frame...
        if end_practice_txt.status == STARTED:
            # update params
            pass
        
        # *continue_txt_12* updates
        
        # if continue_txt_12 is starting this frame...
        if continue_txt_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_12.frameNStart = frameN  # exact frame index
            continue_txt_12.tStart = t  # local t and not account for scr refresh
            continue_txt_12.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_12, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_12.started')
            # update status
            continue_txt_12.status = STARTED
            continue_txt_12.setAutoDraw(True)
        
        # if continue_txt_12 is active this frame...
        if continue_txt_12.status == STARTED:
            # update params
            pass
        # *mouse_14* updates
        
        # if mouse_14 is starting this frame...
        if mouse_14.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_14.frameNStart = frameN  # exact frame index
            mouse_14.tStart = t  # local t and not account for scr refresh
            mouse_14.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_14, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_14.status = STARTED
            mouse_14.mouseClock.reset()
            prevButtonState = mouse_14.getPressed()  # if button is down already this ISN'T a new click
        if mouse_14.status == STARTED:  # only update if started and not finished!
            buttons = mouse_14.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_12, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_14):
                            gotValidClick = True
                            mouse_14.clicked_name.append(obj.name)
                    x, y = mouse_14.getPos()
                    mouse_14.x.append(x)
                    mouse_14.y.append(y)
                    buttons = mouse_14.getPressed()
                    mouse_14.leftButton.append(buttons[0])
                    mouse_14.midButton.append(buttons[1])
                    mouse_14.rightButton.append(buttons[2])
                    mouse_14.time.append(mouse_14.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start_sessionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start_session" ---
    for thisComponent in start_sessionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('start_session.stopped', globalClock.getTime(format='float'))
    # Run 'End Routine' code from start_session_timer
    thisExp.addData('session_start', data.getDateStr())
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_14.x', mouse_14.x)
    thisExp.addData('mouse_14.y', mouse_14.y)
    thisExp.addData('mouse_14.leftButton', mouse_14.leftButton)
    thisExp.addData('mouse_14.midButton', mouse_14.midButton)
    thisExp.addData('mouse_14.rightButton', mouse_14.rightButton)
    thisExp.addData('mouse_14.time', mouse_14.time)
    thisExp.addData('mouse_14.clicked_name', mouse_14.clicked_name)
    thisExp.nextEntry()
    # the Routine "start_session" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('pregen_sequence_' + str(int(expInfo['participant'][-1])) + '.xlsx', selection='0:600'),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "fix" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fix.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from mouse_visible
        win.mouseVisible = False
        mouse.setPos(newPos=(0, 0))
        
        # Run 'Begin Routine' code from set_iti
        import random
        #iti = minDur + (maxDur - minDur) * random.random()
        
        #thisExp.addData("iti", iti)
        # Run 'Begin Routine' code from set_contrast_side
        if eccentricity == -15:
            leftCont = baseContrast+contrastDelta
            rightCont = baseContrast
        elif eccentricity == 15:
            leftCont = baseContrast
            rightCont = baseContrast+contrastDelta
        
        signed_contrast=rightCont-leftCont
        
        # Save all these variables to the log
        thisExp.addData("signed_contrast", signed_contrast)
        thisExp.addData("leftCont", leftCont)
        thisExp.addData("rightCont", rightCont)
        
        #setup for el time triggers
        fixation_2.elOnsetDetected = False
        fixation_2.elOffsetDetected = False
        fixation_2.setColor([1,1,1], colorSpace='rgb')
        fixation_2.setSize((0.75, 0.75))
        # keep track of which components have finished
        fixComponents = [fixation_2]
        for thisComponent in fixComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fix" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from mouse_visible
            mouse.setPos(newPos=(0, 0))
            # Run 'Each Frame' code from set_contrast_side
            if fixation_2.tStartRefresh is not None and not fixation_2.elOnsetDetected:
                el_tracker.sendMessage('fix_cross_ONSET')
                fixation_2.elOnsetDetected = True
            
            if fixation_2.tStopRefresh is not None and fixation_2.tStartRefresh is not None and not fixation_2.elOffsetDetected:
                el_tracker.sendMessage('fix_cross_OFFSET')
                fixation_2.elOffsetDetected = True 
            
            # *fixation_2* updates
            
            # if fixation_2 is starting this frame...
            if fixation_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_2.frameNStart = frameN  # exact frame index
                fixation_2.tStart = t  # local t and not account for scr refresh
                fixation_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_2.started')
                # update status
                fixation_2.status = STARTED
                fixation_2.setAutoDraw(True)
            
            # if fixation_2 is active this frame...
            if fixation_2.status == STARTED:
                # update params
                pass
            
            # if fixation_2 is stopping this frame...
            if fixation_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_2.tStartRefresh + q-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_2.tStop = t  # not accounting for scr refresh
                    fixation_2.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_2.stopped')
                    # update status
                    fixation_2.status = FINISHED
                    fixation_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fix" ---
        for thisComponent in fixComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fix.stopped', globalClock.getTime(format='float'))
        # the Routine "fix" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from dragging_code
        #reset mouse position
        mouse.setPos(newPos=(0, 0))
        
        #set thresholds for response recording for each stim side and corr/wrong
        r_lim_corr = 0.05
        l_lim_corr = -0.05
        r_lim_wrong = 1236
        l_lim_wrong = -1236
        
        #setup for later recording reactio time
        mouserec = mouse.getPos()
        moved = False
        
        #create empty list to later store dot coordinates
        dot_trajectory = []
        
        #setup for el on/offset triggers
        sound_trial_start.elOnsetDetected = False
        sound_trial_start.elOffsetDetected = False
        
        sound_no_resp.elOnsetDetected = False
        sound_no_resp.elOffsetDetected = False
        dot.setPos((618, 0))
        grating_l.setContrast(leftCont)
        grating_l.setPos((-618, 0))
        grating_l.setPhase(random.random()*360)
        grating_r.setContrast(rightCont)
        grating_r.setPos((618, 0))
        grating_r.setPhase(random.random()*360)
        fixation.setColor([1,1,1], colorSpace='rgb')
        fixation.setSize((0.75, 0.75))
        # setup some python lists for storing info about the mouse
        mouse.x = []
        mouse.y = []
        mouse.leftButton = []
        mouse.midButton = []
        mouse.rightButton = []
        mouse.time = []
        gotValidClick = False  # until a click is received
        sound_trial_start.setSound('5000', secs=0.1, hamming=True)
        sound_trial_start.setVolume(0.1, log=False)
        sound_trial_start.seek(0)
        sound_no_resp.setSound('567', secs=0.5, hamming=True)
        sound_no_resp.setVolume(0.1, log=False)
        sound_no_resp.seek(0)
        # This section of EyeLink MarkEventsTrial component code initializes some variables that will help with
        # sending event marking messages, logging Data Viewer (DV) stimulus drawing info, logging DV interest area info,
        # sending DV Target Position Messages, and/or logging DV video frame marking info
        # information
        
        # When we  have multiple trials within one continuous recording we should send a 
        # new TRIALID (for all trials after the first trial of the recording; the first 
        # trial's TRIALID message is sent before recording begins)
        if trial_index > trialIDAtRecordingStart:
            el_tracker.sendMessage('TRIALID %d' % trial_index)
        
        # log trial variables' values to the EDF data file, for details, see Data
        # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
        trialConditionVariablesForEyeLinkLogging = [eccentricity,q,baseContrast,contrastDelta,bias]
        trialConditionVariableNamesForEyeLinkLogging = ['eccentricity', 'q', 'baseContrast', 'contrastDelta', 'bias']
        for i in range(len(trialConditionVariablesForEyeLinkLogging)):
            el_tracker.sendMessage('!V TRIAL_VAR %s %s'% (trialConditionVariableNamesForEyeLinkLogging[i],trialConditionVariablesForEyeLinkLogging[i]))
            #add a brief pause after every 5 messages or so to make sure no messages are missed
            if i % 5 == 0:
                time.sleep(0.001)
        
        # list of all stimulus components whose onset/offset will be marked with messages
        componentsForEyeLinkStimEventMessages = [dot,grating_l,grating_r]
        # create list of all components to be monitored for EyeLink Marking/Messaging
        allStimComponentsForEyeLinkMonitoring = componentsForEyeLinkStimEventMessages# make sure each component is only in the list once
        allStimComponentsForEyeLinkMonitoring = [*set(allStimComponentsForEyeLinkMonitoring)]
        # list of all response components whose onsets need to be marked and values
        # need to be logged
        
        # Initialize stimulus components whose occurence needs to be monitored for event
        # marking, Data Viewer integration, and/or interest area messaging
        # to the EDF (provided they are supported stimulus types)
        for thisComponent in allStimComponentsForEyeLinkMonitoring:
            componentClassString = str(thisComponent.__class__)
            supportedStimType = False
            for stimType in ["Aperture","Text","Dot","Shape","Rect","Grating","Image","MovieStim3","Movie","sound"]:
                if stimType in componentClassString:
                    supportedStimType = True
                    thisComponent.elOnsetDetected = False
                    thisComponent.elOffsetDetected = False
                    if stimType != "sound":
                        thisComponent.elPos = eyelink_pos(thisComponent.pos,[scn_width,scn_height])
                        thisComponent.elSize = eyelink_size(thisComponent.size,[scn_width,scn_height])
                        thisComponent.lastelPos = thisComponent.elPos
                        thisComponent.lastelSize = thisComponent.elSize
                    if stimType == "MovieStim3":
                        thisComponent.componentType = "MovieStim3"
                        thisComponent.elMarkingFrameIndex = -1
                        thisComponent.previousFrameTime = 0
                        thisComponent.firstFramePresented = False
                    elif stimType == "Movie":
                        thisComponent.componentType = "MovieStimWithFrameNum"
                        thisComponent.elMarkingFrameIndex = -1
                        thisComponent.firstFramePresented = False
                    else:
                        thisComponent.componentType = stimType
                    break   
            if not supportedStimType:
                print("WARNING:  Stimulus component type " + str(thisComponent.__class__) + " not supported for EyeLink event marking")
                print("          Event timing messages and/or Data Viewer drawing messages")
                print("          will not be marked for this component")
                print("          Consider marking the component via code component")
                # remove unsupported types from our monitoring lists
                allStimComponentsForEyeLinkMonitoring.remove(thisComponent)
                componentsForEyeLinkStimEventMessages.remove(thisComponent)
                componentsForEyeLinkStimDVDrawingMessages.remove(thisComponent)
                componentsForEyeLinkInterestAreaMessages.remove(thisComponent)
        
        # Send a Data Viewer clear screen command to clear its Trial View window
        # to the window color
        el_tracker.sendMessage('!V CLEAR %d %d %d' % eyelink_color(win.color))
        # create a keyboard instance and reinitialize a kePressNameList, which
        # will store list of key names currently being pressed (to allow Ctrl-C abort)
        kb = keyboard.Keyboard()
        keyPressNameList = []
        eyelinkThisFrameCallOnFlipScheduled = False
        eyelinkLastFlipTime = 0.0
        # keep track of which components have finished
        trialComponents = [dot, grating_l, grating_r, fixation, mouse, sound_trial_start, sound_no_resp, MarkEventsTrial]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from dragging_code
            if sound_trial_start.tStartRefresh is not None and not sound_trial_start.elOnsetDetected:
                el_tracker.sendMessage('signed_contrast %.2f' % signed_contrast)
                el_tracker.sendMessage('stimOn')
                el_tracker.sendMessage('sound_trial_start_ONSET')
                sound_trial_start.elOnsetDetected = True
            
            if sound_trial_start.tStopRefresh is not None and sound_trial_start.tStartRefresh is not None and not sound_trial_start.elOffsetDetected:
                el_tracker.sendMessage('sound_trial_start_OFFSET')
                sound_trial_start.elOffsetDetected = True 
            
            x = mouse.getPos()[0]
            
            if moved == False:
                 mouseloc = mouse.getPos()
                 if mouseloc[0] != mouserec[0] or mouseloc[1] != mouserec[1]:
                      moved = True
                      thisExp.addData('reaction_time',round(t*1000))
                      el_tracker.sendMessage('moveInit')
            
            x_r = x + 618
            x_l = x - 618
            
            grating_r.pos = (x_r,0)
            grating_l.pos = (x_l,0)
            
            if grating_r.overlaps(dot):
                dot.pos = grating_r.pos
            
            elif grating_l.overlaps(dot):
                dot.pos = grating_l.pos
            
            dot_trajectory.append(dot.pos[0])
            
            if eccentricity > 0:
                if grating_r.pos[0] <= r_lim_corr:
                    correct = 1
                    timeout=0
                    response=1
                    el_tracker.sendMessage('response %i' % response)
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
                elif grating_r.pos[0] >= r_lim_wrong:
                    correct = 0
                    timeout=0
                    response=-1
                    el_tracker.sendMessage('response %i' % response)
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
            elif eccentricity < 0:
                if grating_l.pos[0] >= l_lim_corr:
                    correct = 1
                    timeout=0
                    response=-1
                    el_tracker.sendMessage('response %i' % response)
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
                elif grating_l.pos[0] <= l_lim_wrong:
                    correct = 0
                    timeout=0
                    response=1
                    el_tracker.sendMessage('response %i' % response)
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
            
            
            if sound_no_resp.status == STARTED:
                correct = 'NaN'
                timeout = 1
            
            
            if sound_no_resp.tStartRefresh is not None and not sound_no_resp.elOnsetDetected:
                el_tracker.sendMessage('timeout')
                el_tracker.sendMessage('sound_no_resp_ONSET')
                sound_no_resp.elOnsetDetected = True
            
            if sound_no_resp.tStopRefresh is not None and sound_no_resp.tStartRefresh is not None and not sound_no_resp.elOffsetDetected:
                el_tracker.sendMessage('sound_no_resp_OFFSET')
                sound_no_resp.elOffsetDetected = True 
            
            
            # *dot* updates
            
            # if dot is starting this frame...
            if dot.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot.frameNStart = frameN  # exact frame index
                dot.tStart = t  # local t and not account for scr refresh
                dot.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dot.started')
                # update status
                dot.status = STARTED
                dot.setAutoDraw(True)
            
            # if dot is active this frame...
            if dot.status == STARTED:
                # update params
                pass
            
            # if dot is stopping this frame...
            if dot.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dot.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    dot.tStop = t  # not accounting for scr refresh
                    dot.tStopRefresh = tThisFlipGlobal  # on global time
                    dot.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dot.stopped')
                    # update status
                    dot.status = FINISHED
                    dot.setAutoDraw(False)
            
            # *grating_l* updates
            
            # if grating_l is starting this frame...
            if grating_l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                grating_l.frameNStart = frameN  # exact frame index
                grating_l.tStart = t  # local t and not account for scr refresh
                grating_l.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grating_l, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'grating_l.started')
                # update status
                grating_l.status = STARTED
                grating_l.setAutoDraw(True)
            
            # if grating_l is active this frame...
            if grating_l.status == STARTED:
                # update params
                pass
            
            # if grating_l is stopping this frame...
            if grating_l.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grating_l.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    grating_l.tStop = t  # not accounting for scr refresh
                    grating_l.tStopRefresh = tThisFlipGlobal  # on global time
                    grating_l.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grating_l.stopped')
                    # update status
                    grating_l.status = FINISHED
                    grating_l.setAutoDraw(False)
            
            # *grating_r* updates
            
            # if grating_r is starting this frame...
            if grating_r.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                grating_r.frameNStart = frameN  # exact frame index
                grating_r.tStart = t  # local t and not account for scr refresh
                grating_r.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(grating_r, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'grating_r.started')
                # update status
                grating_r.status = STARTED
                grating_r.setAutoDraw(True)
            
            # if grating_r is active this frame...
            if grating_r.status == STARTED:
                # update params
                pass
            
            # if grating_r is stopping this frame...
            if grating_r.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > grating_r.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    grating_r.tStop = t  # not accounting for scr refresh
                    grating_r.tStopRefresh = tThisFlipGlobal  # on global time
                    grating_r.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'grating_r.stopped')
                    # update status
                    grating_r.status = FINISHED
                    grating_r.setAutoDraw(False)
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            # *mouse* updates
            
            # if mouse is starting this frame...
            if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse.frameNStart = frameN  # exact frame index
                mouse.tStart = t  # local t and not account for scr refresh
                mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse.started', t)
                # update status
                mouse.status = STARTED
                prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
            
            # if mouse is stopping this frame...
            if mouse.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > mouse.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    mouse.tStop = t  # not accounting for scr refresh
                    mouse.tStopRefresh = tThisFlipGlobal  # on global time
                    mouse.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('mouse.stopped', t)
                    # update status
                    mouse.status = FINISHED
            if mouse.status == STARTED:  # only update if started and not finished!
                x, y = mouse.getPos()
                mouse.x.append(x)
                mouse.y.append(y)
                buttons = mouse.getPressed()
                mouse.leftButton.append(buttons[0])
                mouse.midButton.append(buttons[1])
                mouse.rightButton.append(buttons[2])
                mouse.time.append(globalClock.getTime())
            
            # if sound_trial_start is starting this frame...
            if sound_trial_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_trial_start.frameNStart = frameN  # exact frame index
                sound_trial_start.tStart = t  # local t and not account for scr refresh
                sound_trial_start.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_trial_start.started', tThisFlipGlobal)
                # update status
                sound_trial_start.status = STARTED
                sound_trial_start.play(when=win)  # sync with win flip
            
            # if sound_trial_start is stopping this frame...
            if sound_trial_start.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_trial_start.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_trial_start.tStop = t  # not accounting for scr refresh
                    sound_trial_start.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_trial_start.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_trial_start.stopped')
                    # update status
                    sound_trial_start.status = FINISHED
                    sound_trial_start.stop()
            # update sound_trial_start status according to whether it's playing
            if sound_trial_start.isPlaying:
                sound_trial_start.status = STARTED
            elif sound_trial_start.isFinished:
                sound_trial_start.status = FINISHED
            
            # if sound_no_resp is starting this frame...
            if sound_no_resp.status == NOT_STARTED and tThisFlip >= 10-frameTolerance:
                # keep track of start time/frame for later
                sound_no_resp.frameNStart = frameN  # exact frame index
                sound_no_resp.tStart = t  # local t and not account for scr refresh
                sound_no_resp.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('sound_no_resp.started', tThisFlipGlobal)
                # update status
                sound_no_resp.status = STARTED
                sound_no_resp.play(when=win)  # sync with win flip
            
            # if sound_no_resp is stopping this frame...
            if sound_no_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_no_resp.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_no_resp.tStop = t  # not accounting for scr refresh
                    sound_no_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    sound_no_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sound_no_resp.stopped')
                    # update status
                    sound_no_resp.status = FINISHED
                    sound_no_resp.stop()
            # update sound_no_resp status according to whether it's playing
            if sound_no_resp.isPlaying:
                sound_no_resp.status = STARTED
            elif sound_no_resp.isFinished:
                sound_no_resp.status = FINISHED
            # This section of EyeLink MarkEventsTrial component code checks whether to send (and sends/logs when appropriate)
            # event marking messages, log Data Viewer (DV) stimulus drawing info, log DV interest area info,
            # send DV Target Position Messages, and/or log DV video frame marking info
            if not eyelinkThisFrameCallOnFlipScheduled:
                # This method, created by the EyeLink MarkEventsTrial component code will get called to handle
                # sending event marking messages, logging Data Viewer (DV) stimulus drawing info, logging DV interest area info,
                # sending DV Target Position Messages, and/or logging DV video frame marking info=information
                win.callOnFlip(eyelink_onFlip_MarkEventsTrial,globalClock,win,scn_width,scn_height,allStimComponentsForEyeLinkMonitoring,\
                    componentsForEyeLinkStimEventMessages)
                eyelinkThisFrameCallOnFlipScheduled = True
            
            # abort the current trial if the tracker is no longer recording
            error = el_tracker.isRecording()
            if error is not pylink.TRIAL_OK:
                el_tracker.sendMessage('tracker_disconnected')
                abort_trial(win,genv)
            
            # check keyboard events for experiment abort key combination
            keyPressList = kb.getKeys(keyList = ['lctrl','rctrl','c'], waitRelease = False, clear = False)
            for keyPress in keyPressList:
                keyPressName = keyPress.name
                if keyPressName not in keyPressNameList:
                    keyPressNameList.append(keyPress.name)
            if ('lctrl' in keyPressNameList or 'rctrl' in keyPressNameList) and 'c' in keyPressNameList:
                el_tracker.sendMessage('terminated_by_user')
                terminate_task(win,genv,edf_file,session_folder,session_identifier)
            #check for key releases
            keyReleaseList = kb.getKeys(keyList = ['lctrl','rctrl','c'], waitRelease = True, clear = False)
            for keyRelease in keyReleaseList:
                keyReleaseName = keyRelease.name
                if keyReleaseName in keyPressNameList:
                    keyPressNameList.remove(keyReleaseName)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial.stopped', globalClock.getTime(format='float'))
        # Run 'End Routine' code from dragging_code
        thisExp.addData("correct", correct)
        thisExp.addData('timeout', timeout)
        thisExp.addData("dot_trajectory", dot_trajectory)
        
        
        # store data for trials (TrialHandler)
        trials.addData('mouse.x', mouse.x)
        trials.addData('mouse.y', mouse.y)
        trials.addData('mouse.leftButton', mouse.leftButton)
        trials.addData('mouse.midButton', mouse.midButton)
        trials.addData('mouse.rightButton', mouse.rightButton)
        trials.addData('mouse.time', mouse.time)
        sound_trial_start.pause()  # ensure sound has stopped at end of Routine
        sound_no_resp.pause()  # ensure sound has stopped at end of Routine
        
        # This section of EyeLink MarkEventsTrial component code does some event cleanup at the end of the routine
        # Go through all stimulus components that need to be checked for event marking,
        #  to see if the trial ended before PsychoPy reported OFFSET detection to mark their offset from trial end
        for thisComponent in componentsForEyeLinkStimEventMessages:
            if thisComponent.elOnsetDetected and not thisComponent.elOffsetDetected:
                # Check if the component had onset but the trial ended before offset
                el_tracker.sendMessage('%s_OFFSET' % (thisComponent.name))
        # Mark the end of the trial for Data Viewer trial parsing
        el_tracker.sendMessage("TRIAL_RESULT 0")
        # Update the EyeLink trial counter
        trial_index = trial_index + 1
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.500000)
        
        # --- Prepare to start Routine "feedback" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('feedback.started', globalClock.getTime(format='float'))
        # Run 'Begin Routine' code from feedback_code
        mouse.setPos(newPos=(0, 0))
        
        el_tracker.sendMessage('feedbackType %s' % correct)
        
        if correct==1:
            fb_sound = "2000.wav"
            fb_sound_dur = 0.2
            fb_dur = 1
            fb_volume = 0.1
            tot_points += 1
            
        elif correct==0:
            fb_sound = "whitenoise.wav"
            fb_sound_dur = 0.5
            fb_dur = 2
            fb_volume = 0.1
        
        else:
            fb_sound = 200
            fb_sound_dur = 0.1
            fb_dur = 1.5
            fb_volume = 0
        
        
        #fixation_3.elOnsetDetected = False
        #fixation_3.elOffsetDetected = False
        
        feedback_sound.elOnsetDetected = False
        feedback_sound.elOffsetDetected = False
        feedback_sound.setSound(fb_sound, secs=fb_sound_dur, hamming=True)
        feedback_sound.setVolume(fb_volume, log=False)
        feedback_sound.seek(0)
        # keep track of which components have finished
        feedbackComponents = [feedback_sound, fixation_3]
        for thisComponent in feedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from feedback_code
            mouse.setPos(newPos=(0, 0))
            
            #if fixation_3.tStartRefresh is not None and not fixation_3.elOnsetDetected:
                #el_tracker.sendMessage('%i %s_ONSET' % (int(round((globalClock.getTime()-fixation_3.tStartRefresh)*1000)),fixation_3.name))
                #fixation_3.elOnsetDetected = True
            
            #if fixation_3.tStopRefresh is not None and fixation_3.tStartRefresh is not None and not fixation_3.elOffsetDetected:
                #el_tracker.sendMessage('%i %s_OFFSET' % (int(round((globalClock.getTime()-fixation_3.tStopRefresh)*1000)),fixation_3.name))
                #fixation_3.elOffsetDetected = True 
            
            if feedback_sound.tStartRefresh is not None and not feedback_sound.elOnsetDetected:
                el_tracker.sendMessage('feedback_sound_ONSET')
                feedback_sound.elOnsetDetected = True
            
            if feedback_sound.tStopRefresh is not None and feedback_sound.tStartRefresh is not None and not feedback_sound.elOffsetDetected:
                el_tracker.sendMessage('feedback_sound_OFFSET')
                feedback_sound.elOffsetDetected = True 
            
            # if feedback_sound is starting this frame...
            if feedback_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_sound.frameNStart = frameN  # exact frame index
                feedback_sound.tStart = t  # local t and not account for scr refresh
                feedback_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('feedback_sound.started', tThisFlipGlobal)
                # update status
                feedback_sound.status = STARTED
                feedback_sound.play(when=win)  # sync with win flip
            
            # if feedback_sound is stopping this frame...
            if feedback_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_sound.tStartRefresh + fb_sound_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_sound.tStop = t  # not accounting for scr refresh
                    feedback_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_sound.stopped')
                    # update status
                    feedback_sound.status = FINISHED
                    feedback_sound.stop()
            # update feedback_sound status according to whether it's playing
            if feedback_sound.isPlaying:
                feedback_sound.status = STARTED
            elif feedback_sound.isFinished:
                feedback_sound.status = FINISHED
            
            # *fixation_3* updates
            
            # if fixation_3 is starting this frame...
            if fixation_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_3.frameNStart = frameN  # exact frame index
                fixation_3.tStart = t  # local t and not account for scr refresh
                fixation_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_3.started')
                # update status
                fixation_3.status = STARTED
                fixation_3.setAutoDraw(True)
            
            # if fixation_3 is active this frame...
            if fixation_3.status == STARTED:
                # update params
                pass
            
            # if fixation_3 is stopping this frame...
            if fixation_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_3.tStartRefresh + fb_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_3.tStop = t  # not accounting for scr refresh
                    fixation_3.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_3.stopped')
                    # update status
                    fixation_3.status = FINISHED
                    fixation_3.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('feedback.stopped', globalClock.getTime(format='float'))
        feedback_sound.pause()  # ensure sound has stopped at end of Routine
        # the Routine "feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "record_delay" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('record_delay.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    record_delayComponents = [blank_txt]
    for thisComponent in record_delayComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "record_delay" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *blank_txt* updates
        
        # if blank_txt is starting this frame...
        if blank_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            blank_txt.frameNStart = frameN  # exact frame index
            blank_txt.tStart = t  # local t and not account for scr refresh
            blank_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(blank_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'blank_txt.started')
            # update status
            blank_txt.status = STARTED
            blank_txt.setAutoDraw(True)
        
        # if blank_txt is active this frame...
        if blank_txt.status == STARTED:
            # update params
            pass
        
        # if blank_txt is stopping this frame...
        if blank_txt.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > blank_txt.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                blank_txt.tStop = t  # not accounting for scr refresh
                blank_txt.tStopRefresh = tThisFlipGlobal  # on global time
                blank_txt.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blank_txt.stopped')
                # update status
                blank_txt.status = FINISHED
                blank_txt.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in record_delayComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "record_delay" ---
    for thisComponent in record_delayComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('record_delay.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "el_stop_rec" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('el_stop_rec.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    el_stop_recComponents = [StopRecord]
    for thisComponent in el_stop_recComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "el_stop_rec" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.001:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in el_stop_recComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "el_stop_rec" ---
    for thisComponent in el_stop_recComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('el_stop_rec.stopped', globalClock.getTime(format='float'))
    # This section of EyeLink StopRecord component code stops recording, sends a trial end (TRIAL_RESULT)
    # message to the EDF, and updates the trial_index variable 
    el_tracker.stopRecording()
    
    # send a 'TRIAL_RESULT' message to mark the end of trial, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    el_tracker.sendMessage('TRIAL_RESULT %d' % 0)
    
    # update the trial counter for the next trial
    trial_index = trial_index + 1
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.001000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "end_task" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('end_task.started', globalClock.getTime(format='float'))
    # Run 'Begin Routine' code from mouse_visible_7
    win.mouseVisible = True
    # Run 'Begin Routine' code from end_session_timer_2
    thisExp.addData('session_end', data.getDateStr())
    
    thisExp.addData('total_points', tot_points)
    # setup some python lists for storing info about the mouse_26
    mouse_26.x = []
    mouse_26.y = []
    mouse_26.leftButton = []
    mouse_26.midButton = []
    mouse_26.rightButton = []
    mouse_26.time = []
    mouse_26.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    end_taskComponents = [end_task_txt, continue_txt_22, mouse_26]
    for thisComponent in end_taskComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_task" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_task_txt* updates
        
        # if end_task_txt is starting this frame...
        if end_task_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_task_txt.frameNStart = frameN  # exact frame index
            end_task_txt.tStart = t  # local t and not account for scr refresh
            end_task_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_task_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_task_txt.started')
            # update status
            end_task_txt.status = STARTED
            end_task_txt.setAutoDraw(True)
        
        # if end_task_txt is active this frame...
        if end_task_txt.status == STARTED:
            # update params
            pass
        
        # *continue_txt_22* updates
        
        # if continue_txt_22 is starting this frame...
        if continue_txt_22.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_22.frameNStart = frameN  # exact frame index
            continue_txt_22.tStart = t  # local t and not account for scr refresh
            continue_txt_22.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_22, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_22.started')
            # update status
            continue_txt_22.status = STARTED
            continue_txt_22.setAutoDraw(True)
        
        # if continue_txt_22 is active this frame...
        if continue_txt_22.status == STARTED:
            # update params
            pass
        # *mouse_26* updates
        
        # if mouse_26 is starting this frame...
        if mouse_26.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_26.frameNStart = frameN  # exact frame index
            mouse_26.tStart = t  # local t and not account for scr refresh
            mouse_26.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_26, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_26.status = STARTED
            mouse_26.mouseClock.reset()
            prevButtonState = mouse_26.getPressed()  # if button is down already this ISN'T a new click
        if mouse_26.status == STARTED:  # only update if started and not finished!
            buttons = mouse_26.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_22, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_26):
                            gotValidClick = True
                            mouse_26.clicked_name.append(obj.name)
                    x, y = mouse_26.getPos()
                    mouse_26.x.append(x)
                    mouse_26.y.append(y)
                    buttons = mouse_26.getPressed()
                    mouse_26.leftButton.append(buttons[0])
                    mouse_26.midButton.append(buttons[1])
                    mouse_26.rightButton.append(buttons[2])
                    mouse_26.time.append(mouse_26.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_taskComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_task" ---
    for thisComponent in end_taskComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('end_task.stopped', globalClock.getTime(format='float'))
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_26.x', mouse_26.x)
    thisExp.addData('mouse_26.y', mouse_26.y)
    thisExp.addData('mouse_26.leftButton', mouse_26.leftButton)
    thisExp.addData('mouse_26.midButton', mouse_26.midButton)
    thisExp.addData('mouse_26.rightButton', mouse_26.rightButton)
    thisExp.addData('mouse_26.time', mouse_26.time)
    thisExp.addData('mouse_26.clicked_name', mouse_26.clicked_name)
    thisExp.nextEntry()
    # the Routine "end_task" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "question1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('question1.started', globalClock.getTime(format='float'))
    textbox1.reset()
    # setup some python lists for storing info about the mouse_27
    mouse_27.x = []
    mouse_27.y = []
    mouse_27.leftButton = []
    mouse_27.midButton = []
    mouse_27.rightButton = []
    mouse_27.time = []
    mouse_27.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    question1Components = [q1_txt, textbox1, continue_txt_23, mouse_27]
    for thisComponent in question1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "question1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *q1_txt* updates
        
        # if q1_txt is starting this frame...
        if q1_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            q1_txt.frameNStart = frameN  # exact frame index
            q1_txt.tStart = t  # local t and not account for scr refresh
            q1_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(q1_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'q1_txt.started')
            # update status
            q1_txt.status = STARTED
            q1_txt.setAutoDraw(True)
        
        # if q1_txt is active this frame...
        if q1_txt.status == STARTED:
            # update params
            pass
        
        # *textbox1* updates
        
        # if textbox1 is starting this frame...
        if textbox1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textbox1.frameNStart = frameN  # exact frame index
            textbox1.tStart = t  # local t and not account for scr refresh
            textbox1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textbox1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textbox1.started')
            # update status
            textbox1.status = STARTED
            textbox1.setAutoDraw(True)
        
        # if textbox1 is active this frame...
        if textbox1.status == STARTED:
            # update params
            pass
        
        # *continue_txt_23* updates
        
        # if continue_txt_23 is starting this frame...
        if continue_txt_23.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_23.frameNStart = frameN  # exact frame index
            continue_txt_23.tStart = t  # local t and not account for scr refresh
            continue_txt_23.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_23, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_23.started')
            # update status
            continue_txt_23.status = STARTED
            continue_txt_23.setAutoDraw(True)
        
        # if continue_txt_23 is active this frame...
        if continue_txt_23.status == STARTED:
            # update params
            pass
        # *mouse_27* updates
        
        # if mouse_27 is starting this frame...
        if mouse_27.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_27.frameNStart = frameN  # exact frame index
            mouse_27.tStart = t  # local t and not account for scr refresh
            mouse_27.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_27, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_27.status = STARTED
            mouse_27.mouseClock.reset()
            prevButtonState = mouse_27.getPressed()  # if button is down already this ISN'T a new click
        if mouse_27.status == STARTED:  # only update if started and not finished!
            buttons = mouse_27.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_23, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_27):
                            gotValidClick = True
                            mouse_27.clicked_name.append(obj.name)
                    x, y = mouse_27.getPos()
                    mouse_27.x.append(x)
                    mouse_27.y.append(y)
                    buttons = mouse_27.getPressed()
                    mouse_27.leftButton.append(buttons[0])
                    mouse_27.midButton.append(buttons[1])
                    mouse_27.rightButton.append(buttons[2])
                    mouse_27.time.append(mouse_27.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in question1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "question1" ---
    for thisComponent in question1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('question1.stopped', globalClock.getTime(format='float'))
    thisExp.addData('textbox1.text',textbox1.text)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_27.x', mouse_27.x)
    thisExp.addData('mouse_27.y', mouse_27.y)
    thisExp.addData('mouse_27.leftButton', mouse_27.leftButton)
    thisExp.addData('mouse_27.midButton', mouse_27.midButton)
    thisExp.addData('mouse_27.rightButton', mouse_27.rightButton)
    thisExp.addData('mouse_27.time', mouse_27.time)
    thisExp.addData('mouse_27.clicked_name', mouse_27.clicked_name)
    thisExp.nextEntry()
    # the Routine "question1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "question2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('question2.started', globalClock.getTime(format='float'))
    textbox2.reset()
    # setup some python lists for storing info about the mouse_28
    mouse_28.x = []
    mouse_28.y = []
    mouse_28.leftButton = []
    mouse_28.midButton = []
    mouse_28.rightButton = []
    mouse_28.time = []
    mouse_28.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    question2Components = [q2_txt, textbox2, continue_txt_24, mouse_28]
    for thisComponent in question2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "question2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *q2_txt* updates
        
        # if q2_txt is starting this frame...
        if q2_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            q2_txt.frameNStart = frameN  # exact frame index
            q2_txt.tStart = t  # local t and not account for scr refresh
            q2_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(q2_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'q2_txt.started')
            # update status
            q2_txt.status = STARTED
            q2_txt.setAutoDraw(True)
        
        # if q2_txt is active this frame...
        if q2_txt.status == STARTED:
            # update params
            pass
        
        # *textbox2* updates
        
        # if textbox2 is starting this frame...
        if textbox2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textbox2.frameNStart = frameN  # exact frame index
            textbox2.tStart = t  # local t and not account for scr refresh
            textbox2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textbox2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textbox2.started')
            # update status
            textbox2.status = STARTED
            textbox2.setAutoDraw(True)
        
        # if textbox2 is active this frame...
        if textbox2.status == STARTED:
            # update params
            pass
        
        # *continue_txt_24* updates
        
        # if continue_txt_24 is starting this frame...
        if continue_txt_24.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_24.frameNStart = frameN  # exact frame index
            continue_txt_24.tStart = t  # local t and not account for scr refresh
            continue_txt_24.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_24, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_24.started')
            # update status
            continue_txt_24.status = STARTED
            continue_txt_24.setAutoDraw(True)
        
        # if continue_txt_24 is active this frame...
        if continue_txt_24.status == STARTED:
            # update params
            pass
        # *mouse_28* updates
        
        # if mouse_28 is starting this frame...
        if mouse_28.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_28.frameNStart = frameN  # exact frame index
            mouse_28.tStart = t  # local t and not account for scr refresh
            mouse_28.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_28, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_28.status = STARTED
            mouse_28.mouseClock.reset()
            prevButtonState = mouse_28.getPressed()  # if button is down already this ISN'T a new click
        if mouse_28.status == STARTED:  # only update if started and not finished!
            buttons = mouse_28.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_24, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_28):
                            gotValidClick = True
                            mouse_28.clicked_name.append(obj.name)
                    x, y = mouse_28.getPos()
                    mouse_28.x.append(x)
                    mouse_28.y.append(y)
                    buttons = mouse_28.getPressed()
                    mouse_28.leftButton.append(buttons[0])
                    mouse_28.midButton.append(buttons[1])
                    mouse_28.rightButton.append(buttons[2])
                    mouse_28.time.append(mouse_28.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in question2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "question2" ---
    for thisComponent in question2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('question2.stopped', globalClock.getTime(format='float'))
    thisExp.addData('textbox2.text',textbox2.text)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_28.x', mouse_28.x)
    thisExp.addData('mouse_28.y', mouse_28.y)
    thisExp.addData('mouse_28.leftButton', mouse_28.leftButton)
    thisExp.addData('mouse_28.midButton', mouse_28.midButton)
    thisExp.addData('mouse_28.rightButton', mouse_28.rightButton)
    thisExp.addData('mouse_28.time', mouse_28.time)
    thisExp.addData('mouse_28.clicked_name', mouse_28.clicked_name)
    thisExp.nextEntry()
    # the Routine "question2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "question3" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('question3.started', globalClock.getTime(format='float'))
    textbox3.reset()
    # setup some python lists for storing info about the mouse_29
    mouse_29.x = []
    mouse_29.y = []
    mouse_29.leftButton = []
    mouse_29.midButton = []
    mouse_29.rightButton = []
    mouse_29.time = []
    mouse_29.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    question3Components = [q3_txt, textbox3, continue_txt_25, mouse_29]
    for thisComponent in question3Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "question3" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *q3_txt* updates
        
        # if q3_txt is starting this frame...
        if q3_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            q3_txt.frameNStart = frameN  # exact frame index
            q3_txt.tStart = t  # local t and not account for scr refresh
            q3_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(q3_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'q3_txt.started')
            # update status
            q3_txt.status = STARTED
            q3_txt.setAutoDraw(True)
        
        # if q3_txt is active this frame...
        if q3_txt.status == STARTED:
            # update params
            pass
        
        # *textbox3* updates
        
        # if textbox3 is starting this frame...
        if textbox3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textbox3.frameNStart = frameN  # exact frame index
            textbox3.tStart = t  # local t and not account for scr refresh
            textbox3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textbox3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textbox3.started')
            # update status
            textbox3.status = STARTED
            textbox3.setAutoDraw(True)
        
        # if textbox3 is active this frame...
        if textbox3.status == STARTED:
            # update params
            pass
        
        # *continue_txt_25* updates
        
        # if continue_txt_25 is starting this frame...
        if continue_txt_25.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_25.frameNStart = frameN  # exact frame index
            continue_txt_25.tStart = t  # local t and not account for scr refresh
            continue_txt_25.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_25, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_25.started')
            # update status
            continue_txt_25.status = STARTED
            continue_txt_25.setAutoDraw(True)
        
        # if continue_txt_25 is active this frame...
        if continue_txt_25.status == STARTED:
            # update params
            pass
        # *mouse_29* updates
        
        # if mouse_29 is starting this frame...
        if mouse_29.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_29.frameNStart = frameN  # exact frame index
            mouse_29.tStart = t  # local t and not account for scr refresh
            mouse_29.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_29, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_29.status = STARTED
            mouse_29.mouseClock.reset()
            prevButtonState = mouse_29.getPressed()  # if button is down already this ISN'T a new click
        if mouse_29.status == STARTED:  # only update if started and not finished!
            buttons = mouse_29.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_25, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_29):
                            gotValidClick = True
                            mouse_29.clicked_name.append(obj.name)
                    x, y = mouse_29.getPos()
                    mouse_29.x.append(x)
                    mouse_29.y.append(y)
                    buttons = mouse_29.getPressed()
                    mouse_29.leftButton.append(buttons[0])
                    mouse_29.midButton.append(buttons[1])
                    mouse_29.rightButton.append(buttons[2])
                    mouse_29.time.append(mouse_29.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in question3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "question3" ---
    for thisComponent in question3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('question3.stopped', globalClock.getTime(format='float'))
    thisExp.addData('textbox3.text',textbox3.text)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_29.x', mouse_29.x)
    thisExp.addData('mouse_29.y', mouse_29.y)
    thisExp.addData('mouse_29.leftButton', mouse_29.leftButton)
    thisExp.addData('mouse_29.midButton', mouse_29.midButton)
    thisExp.addData('mouse_29.rightButton', mouse_29.rightButton)
    thisExp.addData('mouse_29.time', mouse_29.time)
    thisExp.addData('mouse_29.clicked_name', mouse_29.clicked_name)
    thisExp.nextEntry()
    # the Routine "question3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "question4" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('question4.started', globalClock.getTime(format='float'))
    textbox4.reset()
    # setup some python lists for storing info about the mouse_30
    mouse_30.x = []
    mouse_30.y = []
    mouse_30.leftButton = []
    mouse_30.midButton = []
    mouse_30.rightButton = []
    mouse_30.time = []
    mouse_30.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    question4Components = [q4_txt, textbox4, continue_txt_26, mouse_30]
    for thisComponent in question4Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "question4" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *q4_txt* updates
        
        # if q4_txt is starting this frame...
        if q4_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            q4_txt.frameNStart = frameN  # exact frame index
            q4_txt.tStart = t  # local t and not account for scr refresh
            q4_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(q4_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'q4_txt.started')
            # update status
            q4_txt.status = STARTED
            q4_txt.setAutoDraw(True)
        
        # if q4_txt is active this frame...
        if q4_txt.status == STARTED:
            # update params
            pass
        
        # *textbox4* updates
        
        # if textbox4 is starting this frame...
        if textbox4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textbox4.frameNStart = frameN  # exact frame index
            textbox4.tStart = t  # local t and not account for scr refresh
            textbox4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textbox4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textbox4.started')
            # update status
            textbox4.status = STARTED
            textbox4.setAutoDraw(True)
        
        # if textbox4 is active this frame...
        if textbox4.status == STARTED:
            # update params
            pass
        
        # *continue_txt_26* updates
        
        # if continue_txt_26 is starting this frame...
        if continue_txt_26.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_26.frameNStart = frameN  # exact frame index
            continue_txt_26.tStart = t  # local t and not account for scr refresh
            continue_txt_26.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_26, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_26.started')
            # update status
            continue_txt_26.status = STARTED
            continue_txt_26.setAutoDraw(True)
        
        # if continue_txt_26 is active this frame...
        if continue_txt_26.status == STARTED:
            # update params
            pass
        # *mouse_30* updates
        
        # if mouse_30 is starting this frame...
        if mouse_30.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_30.frameNStart = frameN  # exact frame index
            mouse_30.tStart = t  # local t and not account for scr refresh
            mouse_30.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_30, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_30.status = STARTED
            mouse_30.mouseClock.reset()
            prevButtonState = mouse_30.getPressed()  # if button is down already this ISN'T a new click
        if mouse_30.status == STARTED:  # only update if started and not finished!
            buttons = mouse_30.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_26, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_30):
                            gotValidClick = True
                            mouse_30.clicked_name.append(obj.name)
                    x, y = mouse_30.getPos()
                    mouse_30.x.append(x)
                    mouse_30.y.append(y)
                    buttons = mouse_30.getPressed()
                    mouse_30.leftButton.append(buttons[0])
                    mouse_30.midButton.append(buttons[1])
                    mouse_30.rightButton.append(buttons[2])
                    mouse_30.time.append(mouse_30.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in question4Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "question4" ---
    for thisComponent in question4Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('question4.stopped', globalClock.getTime(format='float'))
    thisExp.addData('textbox4.text',textbox4.text)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_30.x', mouse_30.x)
    thisExp.addData('mouse_30.y', mouse_30.y)
    thisExp.addData('mouse_30.leftButton', mouse_30.leftButton)
    thisExp.addData('mouse_30.midButton', mouse_30.midButton)
    thisExp.addData('mouse_30.rightButton', mouse_30.rightButton)
    thisExp.addData('mouse_30.time', mouse_30.time)
    thisExp.addData('mouse_30.clicked_name', mouse_30.clicked_name)
    thisExp.nextEntry()
    # the Routine "question4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "question5" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('question5.started', globalClock.getTime(format='float'))
    textbox5.reset()
    # setup some python lists for storing info about the mouse_33
    mouse_33.x = []
    mouse_33.y = []
    mouse_33.leftButton = []
    mouse_33.midButton = []
    mouse_33.rightButton = []
    mouse_33.time = []
    mouse_33.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    question5Components = [q5_txt, textbox5, continue_txt_29, mouse_33]
    for thisComponent in question5Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "question5" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *q5_txt* updates
        
        # if q5_txt is starting this frame...
        if q5_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            q5_txt.frameNStart = frameN  # exact frame index
            q5_txt.tStart = t  # local t and not account for scr refresh
            q5_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(q5_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'q5_txt.started')
            # update status
            q5_txt.status = STARTED
            q5_txt.setAutoDraw(True)
        
        # if q5_txt is active this frame...
        if q5_txt.status == STARTED:
            # update params
            pass
        
        # *textbox5* updates
        
        # if textbox5 is starting this frame...
        if textbox5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textbox5.frameNStart = frameN  # exact frame index
            textbox5.tStart = t  # local t and not account for scr refresh
            textbox5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textbox5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textbox5.started')
            # update status
            textbox5.status = STARTED
            textbox5.setAutoDraw(True)
        
        # if textbox5 is active this frame...
        if textbox5.status == STARTED:
            # update params
            pass
        
        # *continue_txt_29* updates
        
        # if continue_txt_29 is starting this frame...
        if continue_txt_29.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_29.frameNStart = frameN  # exact frame index
            continue_txt_29.tStart = t  # local t and not account for scr refresh
            continue_txt_29.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_29, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_29.started')
            # update status
            continue_txt_29.status = STARTED
            continue_txt_29.setAutoDraw(True)
        
        # if continue_txt_29 is active this frame...
        if continue_txt_29.status == STARTED:
            # update params
            pass
        # *mouse_33* updates
        
        # if mouse_33 is starting this frame...
        if mouse_33.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_33.frameNStart = frameN  # exact frame index
            mouse_33.tStart = t  # local t and not account for scr refresh
            mouse_33.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_33, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_33.status = STARTED
            mouse_33.mouseClock.reset()
            prevButtonState = mouse_33.getPressed()  # if button is down already this ISN'T a new click
        if mouse_33.status == STARTED:  # only update if started and not finished!
            buttons = mouse_33.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_29, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_33):
                            gotValidClick = True
                            mouse_33.clicked_name.append(obj.name)
                    x, y = mouse_33.getPos()
                    mouse_33.x.append(x)
                    mouse_33.y.append(y)
                    buttons = mouse_33.getPressed()
                    mouse_33.leftButton.append(buttons[0])
                    mouse_33.midButton.append(buttons[1])
                    mouse_33.rightButton.append(buttons[2])
                    mouse_33.time.append(mouse_33.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in question5Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "question5" ---
    for thisComponent in question5Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('question5.stopped', globalClock.getTime(format='float'))
    thisExp.addData('textbox5.text',textbox5.text)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_33.x', mouse_33.x)
    thisExp.addData('mouse_33.y', mouse_33.y)
    thisExp.addData('mouse_33.leftButton', mouse_33.leftButton)
    thisExp.addData('mouse_33.midButton', mouse_33.midButton)
    thisExp.addData('mouse_33.rightButton', mouse_33.rightButton)
    thisExp.addData('mouse_33.time', mouse_33.time)
    thisExp.addData('mouse_33.clicked_name', mouse_33.clicked_name)
    thisExp.nextEntry()
    # the Routine "question5" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "end_exp" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('end_exp.started', globalClock.getTime(format='float'))
    end_exp_txt.setText(f'You have reached the end of the experiment. \n\n You collected {str(tot_points)} points out of 600. \n Thank you very much for participating! \n\n Once you have closed the experiment, you can leave the room. \n The experimenter will then give you more information and answer any questions you may have. \n\n Click Finish to save your data and exit the experiment.')
    # setup some python lists for storing info about the mouse_31
    mouse_31.x = []
    mouse_31.y = []
    mouse_31.leftButton = []
    mouse_31.midButton = []
    mouse_31.rightButton = []
    mouse_31.time = []
    mouse_31.clicked_name = []
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    end_expComponents = [end_exp_txt, continue_txt_27, mouse_31]
    for thisComponent in end_expComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_exp" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_exp_txt* updates
        
        # if end_exp_txt is starting this frame...
        if end_exp_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_exp_txt.frameNStart = frameN  # exact frame index
            end_exp_txt.tStart = t  # local t and not account for scr refresh
            end_exp_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_exp_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_exp_txt.started')
            # update status
            end_exp_txt.status = STARTED
            end_exp_txt.setAutoDraw(True)
        
        # if end_exp_txt is active this frame...
        if end_exp_txt.status == STARTED:
            # update params
            pass
        
        # *continue_txt_27* updates
        
        # if continue_txt_27 is starting this frame...
        if continue_txt_27.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_txt_27.frameNStart = frameN  # exact frame index
            continue_txt_27.tStart = t  # local t and not account for scr refresh
            continue_txt_27.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_txt_27, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_txt_27.started')
            # update status
            continue_txt_27.status = STARTED
            continue_txt_27.setAutoDraw(True)
        
        # if continue_txt_27 is active this frame...
        if continue_txt_27.status == STARTED:
            # update params
            pass
        # *mouse_31* updates
        
        # if mouse_31 is starting this frame...
        if mouse_31.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_31.frameNStart = frameN  # exact frame index
            mouse_31.tStart = t  # local t and not account for scr refresh
            mouse_31.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_31, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_31.status = STARTED
            mouse_31.mouseClock.reset()
            prevButtonState = mouse_31.getPressed()  # if button is down already this ISN'T a new click
        if mouse_31.status == STARTED:  # only update if started and not finished!
            buttons = mouse_31.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    # check if the mouse was inside our 'clickable' objects
                    gotValidClick = False
                    clickableList = environmenttools.getFromNames(continue_txt_27, namespace=locals())
                    for obj in clickableList:
                        # is this object clicked on?
                        if obj.contains(mouse_31):
                            gotValidClick = True
                            mouse_31.clicked_name.append(obj.name)
                    x, y = mouse_31.getPos()
                    mouse_31.x.append(x)
                    mouse_31.y.append(y)
                    buttons = mouse_31.getPressed()
                    mouse_31.leftButton.append(buttons[0])
                    mouse_31.midButton.append(buttons[1])
                    mouse_31.rightButton.append(buttons[2])
                    mouse_31.time.append(mouse_31.mouseClock.getTime())
                    if gotValidClick:
                        continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_expComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_exp" ---
    for thisComponent in end_expComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('end_exp.stopped', globalClock.getTime(format='float'))
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_31.x', mouse_31.x)
    thisExp.addData('mouse_31.y', mouse_31.y)
    thisExp.addData('mouse_31.leftButton', mouse_31.leftButton)
    thisExp.addData('mouse_31.midButton', mouse_31.midButton)
    thisExp.addData('mouse_31.rightButton', mouse_31.rightButton)
    thisExp.addData('mouse_31.time', mouse_31.time)
    thisExp.addData('mouse_31.clicked_name', mouse_31.clicked_name)
    thisExp.nextEntry()
    # the Routine "end_exp" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # This section of the Initialize component calls the 
    # terminate_task helper function to get the EDF file and close the connection
    # to the Host PC
    
    # Disconnect, download the EDF file, then terminate the task
    terminate_task(win,genv,edf_file,session_folder,session_identifier)
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
