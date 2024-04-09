#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on Tue Apr  9 13:58:20 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
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

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'pedestal_cursor_ibl'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
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
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/camillaucheomaenwereuzor/Desktop/RA IBL task/mouse version prova/WIN_pedestal_cursor_ibl_lastrun.py',
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
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
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
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1440, 900], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='labMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='deg'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'deg'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
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
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
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
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
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
    inputs : dict
        Dictionary of input devices by name.
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
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
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
        text='Welcome to the experiment!\n\nPlease position yourself comfortably in front of the computer, making sure that you can comfortably reach the keyboard. Try to keep this position throughout the experiment.\n\nPress space to continue',
        font='Arial',
        pos=(0, 0), height=0.75, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    space_continue = keyboard.Keyboard()
    
    # --- Initialize components for Routine "fix" ---
    # Run 'Begin Experiment' code from set_iti
    iti = 0
    maxDur = 0.7
    minDur = 0.4
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
    fixation = visual.ImageStim(
        win=win,
        name='fixation', units='deg', 
        image='fixation_object.png', mask='circle', anchor='center',
        ori=0.0, pos=(0, 0), size=1.0,
        color='white', colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    dot = visual.ShapeStim(
        win=win, name='dot',
        size=(0.5, 0.5), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[0,0,0], fillColor=[0,0,0],
        opacity=None, depth=-2.0, interpolate=True)
    grating_l = visual.GratingStim(
        win=win, name='grating_l',units='deg', 
        tex='sin', mask='gauss', anchor='center',
        ori=0.0, pos=[0,0], size=(15, 15), sf=0.3, phase=1.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-3.0)
    grating_r = visual.GratingStim(
        win=win, name='grating_r',
        tex='sin', mask='gauss', anchor='center',
        ori=0.0, pos=[0,0], size=(15, 15), sf=0.3, phase=1.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-4.0)
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    sound_trial_start = sound.Sound('5000', secs=0.1, stereo=True, hamming=True,
        name='sound_trial_start')
    sound_trial_start.setVolume(0.1)
    sound_no_resp = sound.Sound('567', secs=0.5, stereo=True, hamming=True,
        name='sound_no_resp')
    sound_no_resp.setVolume(0.1)
    
    # --- Initialize components for Routine "feedback" ---
    # Run 'Begin Experiment' code from feedback_code
    fb_sound = 100
    fb_dur = 10
    feedback_sound = sound.Sound(fb_sound, secs=-1, stereo=True, hamming=True,
        name='feedback_sound')
    feedback_sound.setVolume(0.1)
    fixation_obj_2 = visual.ImageStim(
        win=win,
        name='fixation_obj_2', 
        image='fixation_object.png', mask='circle', anchor='center',
        ori=0.0, pos=(0, 0), size=(0.025, 0.025),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "block_break" ---
    # Run 'Begin Experiment' code from block_counter
    nBlocks=0
    break_text = visual.TextStim(win=win, name='break_text',
        text='You can take a quick break to rest your eyes.\n\nThe experiment will resume automatically in',
        font='Arial',
        pos=(0, 0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    countdown = visual.TextStim(win=win, name='countdown',
        text='',
        font='Arial',
        pos=(0, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "fix" ---
    # Run 'Begin Experiment' code from set_iti
    iti = 0
    maxDur = 0.7
    minDur = 0.4
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
    fixation = visual.ImageStim(
        win=win,
        name='fixation', units='deg', 
        image='fixation_object.png', mask='circle', anchor='center',
        ori=0.0, pos=(0, 0), size=1.0,
        color='white', colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    dot = visual.ShapeStim(
        win=win, name='dot',
        size=(0.5, 0.5), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[0,0,0], fillColor=[0,0,0],
        opacity=None, depth=-2.0, interpolate=True)
    grating_l = visual.GratingStim(
        win=win, name='grating_l',units='deg', 
        tex='sin', mask='gauss', anchor='center',
        ori=0.0, pos=[0,0], size=(15, 15), sf=0.3, phase=1.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-3.0)
    grating_r = visual.GratingStim(
        win=win, name='grating_r',
        tex='sin', mask='gauss', anchor='center',
        ori=0.0, pos=[0,0], size=(15, 15), sf=0.3, phase=1.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-4.0)
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    sound_trial_start = sound.Sound('5000', secs=0.1, stereo=True, hamming=True,
        name='sound_trial_start')
    sound_trial_start.setVolume(0.1)
    sound_no_resp = sound.Sound('567', secs=0.5, stereo=True, hamming=True,
        name='sound_no_resp')
    sound_no_resp.setVolume(0.1)
    
    # --- Initialize components for Routine "feedback" ---
    # Run 'Begin Experiment' code from feedback_code
    fb_sound = 100
    fb_dur = 10
    feedback_sound = sound.Sound(fb_sound, secs=-1, stereo=True, hamming=True,
        name='feedback_sound')
    feedback_sound.setVolume(0.1)
    fixation_obj_2 = visual.ImageStim(
        win=win,
        name='fixation_obj_2', 
        image='fixation_object.png', mask='circle', anchor='center',
        ori=0.0, pos=(0, 0), size=(0.025, 0.025),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "block_break" ---
    # Run 'Begin Experiment' code from block_counter
    nBlocks=0
    break_text = visual.TextStim(win=win, name='break_text',
        text='You can take a quick break to rest your eyes.\n\nThe experiment will resume automatically in',
        font='Arial',
        pos=(0, 0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    countdown = visual.TextStim(win=win, name='countdown',
        text='',
        font='Arial',
        pos=(0, -0.1), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "outro" ---
    outro_text = visual.TextStim(win=win, name='outro_text',
        text="That's the end of the experiment.\n\nThank you for participating!\nYou can collect your things and exit the room. The experiment leader will give you more information about the study.\n\nPress space to finish",
        font='Arial',
        pos=(0, 0), height=1.0, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    space_continue_7 = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('welcome.started', globalClock.getTime())
    space_continue.keys = []
    space_continue.rt = []
    _space_continue_allKeys = []
    # Run 'Begin Routine' code from mouse_visible_2
    win.mouseVisible = False
    # keep track of which components have finished
    welcomeComponents = [welcome_position, space_continue]
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
        if welcome_position.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
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
        
        # *space_continue* updates
        waitOnFlip = False
        
        # if space_continue is starting this frame...
        if space_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            space_continue.frameNStart = frameN  # exact frame index
            space_continue.tStart = t  # local t and not account for scr refresh
            space_continue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(space_continue, 'tStartRefresh')  # time at next scr refresh
            # update status
            space_continue.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(space_continue.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(space_continue.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if space_continue.status == STARTED and not waitOnFlip:
            theseKeys = space_continue.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _space_continue_allKeys.extend(theseKeys)
            if len(_space_continue_allKeys):
                space_continue.keys = _space_continue_allKeys[-1].name  # just the last key pressed
                space_continue.rt = _space_continue_allKeys[-1].rt
                space_continue.duration = _space_continue_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
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
    thisExp.addData('welcome.stopped', globalClock.getTime())
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('pregen_params.xlsx', selection='2:10'),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
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
        thisExp.addData('fix.started', globalClock.getTime())
        # Run 'Begin Routine' code from mouse_visible
        win.mouseVisible = False
        # Run 'Begin Routine' code from set_iti
        import random
        iti = minDur + (maxDur - minDur) * random.random()
        
        thisExp.addData("iti", iti)
        # Run 'Begin Routine' code from set_contrast_side
        #import random
        #deltaSide=(1, -1)[random.random()>biasPoint] # Using a ternary operator to pick a side at random (+ bias if required). 
        # A bias point of .2 means that the stimulus has an 80% chance of being assigned a -1 value (it's most likely to be on the left).
        # The operator evaluates the thing on the right as 0 or 1 (true or false)
        # and then picks the 0th or 1th thing from the preceding list.
        
        #leftCont=baseContrast+contrastDelta*(deltaSide==-1)
        #rightCont=baseContrast+contrastDelta*(deltaSide==1)
        if eccentricity == -15:
            leftCont = baseContrast+contrastDelta
            rightCont = baseContrast
        elif eccentricity == 15:
            leftCont = baseContrast
            rightCont = basecontrast+contrastDelta
        
        signed_contrast=rightCont-leftCont
        
        #if leftCont == rightCont:
        #    eccentricity = -15
        #elif leftCont > rightCont:
        #    eccentricity = -15
        #elif rightCont > leftCont:
        #    eccentricity = 15
        
        # Save all these variables to the log
        thisExp.addData("signed_contrast", signed_contrast)
        thisExp.addData("leftCont", leftCont)
        thisExp.addData("rightCont", rightCont)
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
            
            # *fixation_2* updates
            
            # if fixation_2 is starting this frame...
            if fixation_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_2.frameNStart = frameN  # exact frame index
                fixation_2.tStart = t  # local t and not account for scr refresh
                fixation_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_2, 'tStartRefresh')  # time at next scr refresh
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
                if tThisFlipGlobal > fixation_2.tStartRefresh + iti-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_2.tStop = t  # not accounting for scr refresh
                    fixation_2.frameNStop = frameN  # exact frame index
                    # update status
                    fixation_2.status = FINISHED
                    fixation_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
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
        thisExp.addData('fix.stopped', globalClock.getTime())
        # the Routine "fix" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime())
        # Run 'Begin Routine' code from dragging_code
        mouse.setPos(newPos=(0, 0))
        
        r_lim_corr = 0.05
        l_lim_corr = -0.05
        r_lim_wrong = 30
        l_lim_wrong = -30
        
        mouserec = mouse.getPos()
        moved = False
        fixation.setColor([0,0,0], colorSpace='rgb')
        fixation.setSize((0.75, 0.75))
        dot.setPos((eccentricity, 0))
        grating_l.setContrast(leftCont)
        grating_l.setPos((-35, 0))
        grating_l.setPhase(random.random()*360)
        grating_r.setContrast(rightCont)
        grating_r.setPos((35, 0))
        grating_r.setPhase(random.random()*360)
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
        # keep track of which components have finished
        trialComponents = [fixation, dot, grating_l, grating_r, mouse, sound_trial_start, sound_no_resp]
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
            x = mouse.getPos()[0]
            #y = mouse.getPos()[1]
            
            if moved == False:
                 mouseloc = mouse.getPos()
                 if mouseloc[0] != mouserec[0] or mouseloc[1] != mouserec[1]:
                      moved = True
                      thisExp.addData('reaction_time',round(t*1000))
            
            x_r = x + 15
            x_l = x - 15
            
            grating_r.pos = (x_r,0)
            grating_l.pos = (x_l,0)
            
            if grating_r.overlaps(dot):
                dot.pos = grating_r.pos
            elif grating_l.overlaps(dot):
                dot.pos = grating_l.pos
            
            
            if eccentricity > 0:
                if grating_r.pos[0] <= r_lim_corr:
                    correct = 1
                    timeout='n'
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
                elif grating_r.pos[0] >= r_lim_wrong:
                    correct = 0
                    timeout='n'
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
            elif eccentricity < 0:
                if grating_l.pos[0] >= l_lim_corr:
                    correct = 1
                    timeout='n'
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
                elif grating_l.pos[0] <= l_lim_wrong:
                    correct = 0
                    timeout='n'
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
            
            #if fixation.overlaps(dot):
            #    correct = 1
            #    timeout='n'
            #    thisExp.addData("response_time", round(t*1000))
            #    continueRoutine=False
            #elif dot.pos[0] > r_lim_wrong or dot.pos[0] < l_lim_wrong:
            #    correct = 0
            #    timeout='n'
            #    thisExp.addData("response_time", round(t*1000))
            #    continueRoutine=False
            
            
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
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
                    fixation.frameNStop = frameN  # exact frame index
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *dot* updates
            
            # if dot is starting this frame...
            if dot.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot.frameNStart = frameN  # exact frame index
                dot.tStart = t  # local t and not account for scr refresh
                dot.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot, 'tStartRefresh')  # time at next scr refresh
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
                    dot.frameNStop = frameN  # exact frame index
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
                    grating_l.frameNStop = frameN  # exact frame index
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
                    grating_r.frameNStop = frameN  # exact frame index
                    # update status
                    grating_r.status = FINISHED
                    grating_r.setAutoDraw(False)
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
                mouse.mouseClock.reset()
                prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
            
            # if mouse is stopping this frame...
            if mouse.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > mouse.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    mouse.tStop = t  # not accounting for scr refresh
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
                mouse.time.append(mouse.mouseClock.getTime())
            
            # if sound_trial_start is starting this frame...
            if sound_trial_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_trial_start.frameNStart = frameN  # exact frame index
                sound_trial_start.tStart = t  # local t and not account for scr refresh
                sound_trial_start.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                sound_trial_start.status = STARTED
                sound_trial_start.play(when=win)  # sync with win flip
            
            # if sound_trial_start is stopping this frame...
            if sound_trial_start.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_trial_start.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_trial_start.tStop = t  # not accounting for scr refresh
                    sound_trial_start.frameNStop = frameN  # exact frame index
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
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
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
        thisExp.addData('trial.stopped', globalClock.getTime())
        # Run 'End Routine' code from dragging_code
        if correct!=1 and correct!=0:
            timeout='y'
            correct='NaN'
        
        thisExp.addData("correct", correct)
        thisExp.addData('timeout', timeout)
        # store data for trials (TrialHandler)
        trials.addData('mouse.x', mouse.x)
        trials.addData('mouse.y', mouse.y)
        trials.addData('mouse.leftButton', mouse.leftButton)
        trials.addData('mouse.midButton', mouse.midButton)
        trials.addData('mouse.rightButton', mouse.rightButton)
        trials.addData('mouse.time', mouse.time)
        sound_trial_start.pause()  # ensure sound has stopped at end of Routine
        sound_no_resp.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.500000)
        
        # --- Prepare to start Routine "feedback" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('feedback.started', globalClock.getTime())
        # Run 'Begin Routine' code from feedback_code
        if correct==1:
            fb_sound = 2000
            fb_sound_dur = 0.2
            fb_dur = 1
            
        else:
            fb_sound = 567
            fb_sound_dur = 0.5
            fb_dur = 2
        feedback_sound.setSound(fb_sound, secs=fb_sound_dur, hamming=True)
        feedback_sound.setVolume(0.1, log=False)
        feedback_sound.seek(0)
        # keep track of which components have finished
        feedbackComponents = [feedback_sound, fixation_obj_2]
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
            
            # *fixation_obj_2* updates
            
            # if fixation_obj_2 is starting this frame...
            if fixation_obj_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_obj_2.frameNStart = frameN  # exact frame index
                fixation_obj_2.tStart = t  # local t and not account for scr refresh
                fixation_obj_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_obj_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_obj_2.started')
                # update status
                fixation_obj_2.status = STARTED
                fixation_obj_2.setAutoDraw(True)
            
            # if fixation_obj_2 is active this frame...
            if fixation_obj_2.status == STARTED:
                # update params
                pass
            
            # if fixation_obj_2 is stopping this frame...
            if fixation_obj_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_obj_2.tStartRefresh + fb_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_obj_2.tStop = t  # not accounting for scr refresh
                    fixation_obj_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_obj_2.stopped')
                    # update status
                    fixation_obj_2.status = FINISHED
                    fixation_obj_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
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
        thisExp.addData('feedback.stopped', globalClock.getTime())
        feedback_sound.pause()  # ensure sound has stopped at end of Routine
        # the Routine "feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "block_break" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('block_break.started', globalClock.getTime())
    # keep track of which components have finished
    block_breakComponents = [break_text, countdown]
    for thisComponent in block_breakComponents:
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
    
    # --- Run Routine "block_break" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from block_counter
        #if nBlocks==1:
        #    continueRoutine=False
        
        # *break_text* updates
        
        # if break_text is starting this frame...
        if break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            break_text.frameNStart = frameN  # exact frame index
            break_text.tStart = t  # local t and not account for scr refresh
            break_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(break_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'break_text.started')
            # update status
            break_text.status = STARTED
            break_text.setAutoDraw(True)
        
        # if break_text is active this frame...
        if break_text.status == STARTED:
            # update params
            pass
        
        # if break_text is stopping this frame...
        if break_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > break_text.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                break_text.tStop = t  # not accounting for scr refresh
                break_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_text.stopped')
                # update status
                break_text.status = FINISHED
                break_text.setAutoDraw(False)
        
        # *countdown* updates
        
        # if countdown is starting this frame...
        if countdown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            countdown.frameNStart = frameN  # exact frame index
            countdown.tStart = t  # local t and not account for scr refresh
            countdown.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(countdown, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'countdown.started')
            # update status
            countdown.status = STARTED
            countdown.setAutoDraw(True)
        
        # if countdown is active this frame...
        if countdown.status == STARTED:
            # update params
            countdown.setText(int(round(10 - t, 3)), log=False)
        
        # if countdown is stopping this frame...
        if countdown.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > countdown.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                countdown.tStop = t  # not accounting for scr refresh
                countdown.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'countdown.stopped')
                # update status
                countdown.status = FINISHED
                countdown.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in block_breakComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "block_break" ---
    for thisComponent in block_breakComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('block_break.stopped', globalClock.getTime())
    # Run 'End Routine' code from block_counter
    nBlocks+=1
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.000000)
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('pregen_params.xlsx', selection='101:200'),
        seed=None, name='trials_2')
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # --- Prepare to start Routine "fix" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fix.started', globalClock.getTime())
        # Run 'Begin Routine' code from mouse_visible
        win.mouseVisible = False
        # Run 'Begin Routine' code from set_iti
        import random
        iti = minDur + (maxDur - minDur) * random.random()
        
        thisExp.addData("iti", iti)
        # Run 'Begin Routine' code from set_contrast_side
        #import random
        #deltaSide=(1, -1)[random.random()>biasPoint] # Using a ternary operator to pick a side at random (+ bias if required). 
        # A bias point of .2 means that the stimulus has an 80% chance of being assigned a -1 value (it's most likely to be on the left).
        # The operator evaluates the thing on the right as 0 or 1 (true or false)
        # and then picks the 0th or 1th thing from the preceding list.
        
        #leftCont=baseContrast+contrastDelta*(deltaSide==-1)
        #rightCont=baseContrast+contrastDelta*(deltaSide==1)
        if eccentricity == -15:
            leftCont = baseContrast+contrastDelta
            rightCont = baseContrast
        elif eccentricity == 15:
            leftCont = baseContrast
            rightCont = basecontrast+contrastDelta
        
        signed_contrast=rightCont-leftCont
        
        #if leftCont == rightCont:
        #    eccentricity = -15
        #elif leftCont > rightCont:
        #    eccentricity = -15
        #elif rightCont > leftCont:
        #    eccentricity = 15
        
        # Save all these variables to the log
        thisExp.addData("signed_contrast", signed_contrast)
        thisExp.addData("leftCont", leftCont)
        thisExp.addData("rightCont", rightCont)
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
            
            # *fixation_2* updates
            
            # if fixation_2 is starting this frame...
            if fixation_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_2.frameNStart = frameN  # exact frame index
                fixation_2.tStart = t  # local t and not account for scr refresh
                fixation_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_2, 'tStartRefresh')  # time at next scr refresh
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
                if tThisFlipGlobal > fixation_2.tStartRefresh + iti-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_2.tStop = t  # not accounting for scr refresh
                    fixation_2.frameNStop = frameN  # exact frame index
                    # update status
                    fixation_2.status = FINISHED
                    fixation_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
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
        thisExp.addData('fix.stopped', globalClock.getTime())
        # the Routine "fix" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime())
        # Run 'Begin Routine' code from dragging_code
        mouse.setPos(newPos=(0, 0))
        
        r_lim_corr = 0.05
        l_lim_corr = -0.05
        r_lim_wrong = 30
        l_lim_wrong = -30
        
        mouserec = mouse.getPos()
        moved = False
        fixation.setColor([0,0,0], colorSpace='rgb')
        fixation.setSize((0.75, 0.75))
        dot.setPos((eccentricity, 0))
        grating_l.setContrast(leftCont)
        grating_l.setPos((-35, 0))
        grating_l.setPhase(random.random()*360)
        grating_r.setContrast(rightCont)
        grating_r.setPos((35, 0))
        grating_r.setPhase(random.random()*360)
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
        # keep track of which components have finished
        trialComponents = [fixation, dot, grating_l, grating_r, mouse, sound_trial_start, sound_no_resp]
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
            x = mouse.getPos()[0]
            #y = mouse.getPos()[1]
            
            if moved == False:
                 mouseloc = mouse.getPos()
                 if mouseloc[0] != mouserec[0] or mouseloc[1] != mouserec[1]:
                      moved = True
                      thisExp.addData('reaction_time',round(t*1000))
            
            x_r = x + 15
            x_l = x - 15
            
            grating_r.pos = (x_r,0)
            grating_l.pos = (x_l,0)
            
            if grating_r.overlaps(dot):
                dot.pos = grating_r.pos
            elif grating_l.overlaps(dot):
                dot.pos = grating_l.pos
            
            
            if eccentricity > 0:
                if grating_r.pos[0] <= r_lim_corr:
                    correct = 1
                    timeout='n'
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
                elif grating_r.pos[0] >= r_lim_wrong:
                    correct = 0
                    timeout='n'
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
            elif eccentricity < 0:
                if grating_l.pos[0] >= l_lim_corr:
                    correct = 1
                    timeout='n'
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
                elif grating_l.pos[0] <= l_lim_wrong:
                    correct = 0
                    timeout='n'
                    thisExp.addData("response_time", round(t*1000))
                    continueRoutine=False
            
            #if fixation.overlaps(dot):
            #    correct = 1
            #    timeout='n'
            #    thisExp.addData("response_time", round(t*1000))
            #    continueRoutine=False
            #elif dot.pos[0] > r_lim_wrong or dot.pos[0] < l_lim_wrong:
            #    correct = 0
            #    timeout='n'
            #    thisExp.addData("response_time", round(t*1000))
            #    continueRoutine=False
            
            
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
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
                    fixation.frameNStop = frameN  # exact frame index
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *dot* updates
            
            # if dot is starting this frame...
            if dot.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dot.frameNStart = frameN  # exact frame index
                dot.tStart = t  # local t and not account for scr refresh
                dot.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dot, 'tStartRefresh')  # time at next scr refresh
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
                    dot.frameNStop = frameN  # exact frame index
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
                    grating_l.frameNStop = frameN  # exact frame index
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
                    grating_r.frameNStop = frameN  # exact frame index
                    # update status
                    grating_r.status = FINISHED
                    grating_r.setAutoDraw(False)
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
                mouse.mouseClock.reset()
                prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
            
            # if mouse is stopping this frame...
            if mouse.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > mouse.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    mouse.tStop = t  # not accounting for scr refresh
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
                mouse.time.append(mouse.mouseClock.getTime())
            
            # if sound_trial_start is starting this frame...
            if sound_trial_start.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                sound_trial_start.frameNStart = frameN  # exact frame index
                sound_trial_start.tStart = t  # local t and not account for scr refresh
                sound_trial_start.tStartRefresh = tThisFlipGlobal  # on global time
                # update status
                sound_trial_start.status = STARTED
                sound_trial_start.play(when=win)  # sync with win flip
            
            # if sound_trial_start is stopping this frame...
            if sound_trial_start.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > sound_trial_start.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    sound_trial_start.tStop = t  # not accounting for scr refresh
                    sound_trial_start.frameNStop = frameN  # exact frame index
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
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
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
        thisExp.addData('trial.stopped', globalClock.getTime())
        # Run 'End Routine' code from dragging_code
        if correct!=1 and correct!=0:
            timeout='y'
            correct='NaN'
        
        thisExp.addData("correct", correct)
        thisExp.addData('timeout', timeout)
        # store data for trials_2 (TrialHandler)
        trials_2.addData('mouse.x', mouse.x)
        trials_2.addData('mouse.y', mouse.y)
        trials_2.addData('mouse.leftButton', mouse.leftButton)
        trials_2.addData('mouse.midButton', mouse.midButton)
        trials_2.addData('mouse.rightButton', mouse.rightButton)
        trials_2.addData('mouse.time', mouse.time)
        sound_trial_start.pause()  # ensure sound has stopped at end of Routine
        sound_no_resp.pause()  # ensure sound has stopped at end of Routine
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.500000)
        
        # --- Prepare to start Routine "feedback" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('feedback.started', globalClock.getTime())
        # Run 'Begin Routine' code from feedback_code
        if correct==1:
            fb_sound = 2000
            fb_sound_dur = 0.2
            fb_dur = 1
            
        else:
            fb_sound = 567
            fb_sound_dur = 0.5
            fb_dur = 2
        feedback_sound.setSound(fb_sound, secs=fb_sound_dur, hamming=True)
        feedback_sound.setVolume(0.1, log=False)
        feedback_sound.seek(0)
        # keep track of which components have finished
        feedbackComponents = [feedback_sound, fixation_obj_2]
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
            
            # *fixation_obj_2* updates
            
            # if fixation_obj_2 is starting this frame...
            if fixation_obj_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_obj_2.frameNStart = frameN  # exact frame index
                fixation_obj_2.tStart = t  # local t and not account for scr refresh
                fixation_obj_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_obj_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_obj_2.started')
                # update status
                fixation_obj_2.status = STARTED
                fixation_obj_2.setAutoDraw(True)
            
            # if fixation_obj_2 is active this frame...
            if fixation_obj_2.status == STARTED:
                # update params
                pass
            
            # if fixation_obj_2 is stopping this frame...
            if fixation_obj_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_obj_2.tStartRefresh + fb_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_obj_2.tStop = t  # not accounting for scr refresh
                    fixation_obj_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_obj_2.stopped')
                    # update status
                    fixation_obj_2.status = FINISHED
                    fixation_obj_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
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
        thisExp.addData('feedback.stopped', globalClock.getTime())
        feedback_sound.pause()  # ensure sound has stopped at end of Routine
        # the Routine "feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_2'
    
    
    # --- Prepare to start Routine "block_break" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('block_break.started', globalClock.getTime())
    # keep track of which components have finished
    block_breakComponents = [break_text, countdown]
    for thisComponent in block_breakComponents:
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
    
    # --- Run Routine "block_break" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from block_counter
        #if nBlocks==1:
        #    continueRoutine=False
        
        # *break_text* updates
        
        # if break_text is starting this frame...
        if break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            break_text.frameNStart = frameN  # exact frame index
            break_text.tStart = t  # local t and not account for scr refresh
            break_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(break_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'break_text.started')
            # update status
            break_text.status = STARTED
            break_text.setAutoDraw(True)
        
        # if break_text is active this frame...
        if break_text.status == STARTED:
            # update params
            pass
        
        # if break_text is stopping this frame...
        if break_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > break_text.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                break_text.tStop = t  # not accounting for scr refresh
                break_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_text.stopped')
                # update status
                break_text.status = FINISHED
                break_text.setAutoDraw(False)
        
        # *countdown* updates
        
        # if countdown is starting this frame...
        if countdown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            countdown.frameNStart = frameN  # exact frame index
            countdown.tStart = t  # local t and not account for scr refresh
            countdown.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(countdown, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'countdown.started')
            # update status
            countdown.status = STARTED
            countdown.setAutoDraw(True)
        
        # if countdown is active this frame...
        if countdown.status == STARTED:
            # update params
            countdown.setText(int(round(10 - t, 3)), log=False)
        
        # if countdown is stopping this frame...
        if countdown.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > countdown.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                countdown.tStop = t  # not accounting for scr refresh
                countdown.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'countdown.stopped')
                # update status
                countdown.status = FINISHED
                countdown.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in block_breakComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "block_break" ---
    for thisComponent in block_breakComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('block_break.stopped', globalClock.getTime())
    # Run 'End Routine' code from block_counter
    nBlocks+=1
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.000000)
    
    # --- Prepare to start Routine "outro" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('outro.started', globalClock.getTime())
    space_continue_7.keys = []
    space_continue_7.rt = []
    _space_continue_7_allKeys = []
    # keep track of which components have finished
    outroComponents = [outro_text, space_continue_7]
    for thisComponent in outroComponents:
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
    
    # --- Run Routine "outro" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *outro_text* updates
        
        # if outro_text is starting this frame...
        if outro_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            outro_text.frameNStart = frameN  # exact frame index
            outro_text.tStart = t  # local t and not account for scr refresh
            outro_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(outro_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'outro_text.started')
            # update status
            outro_text.status = STARTED
            outro_text.setAutoDraw(True)
        
        # if outro_text is active this frame...
        if outro_text.status == STARTED:
            # update params
            pass
        
        # *space_continue_7* updates
        waitOnFlip = False
        
        # if space_continue_7 is starting this frame...
        if space_continue_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            space_continue_7.frameNStart = frameN  # exact frame index
            space_continue_7.tStart = t  # local t and not account for scr refresh
            space_continue_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(space_continue_7, 'tStartRefresh')  # time at next scr refresh
            # update status
            space_continue_7.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(space_continue_7.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(space_continue_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if space_continue_7.status == STARTED and not waitOnFlip:
            theseKeys = space_continue_7.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _space_continue_7_allKeys.extend(theseKeys)
            if len(_space_continue_7_allKeys):
                space_continue_7.keys = _space_continue_7_allKeys[-1].name  # just the last key pressed
                space_continue_7.rt = _space_continue_7_allKeys[-1].rt
                space_continue_7.duration = _space_continue_7_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in outroComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "outro" ---
    for thisComponent in outroComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('outro.stopped', globalClock.getTime())
    # the Routine "outro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


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


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
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
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
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
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
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
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
