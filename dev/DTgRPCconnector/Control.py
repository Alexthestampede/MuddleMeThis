# FlatBuffer Control table for ControlNet support
# Matches config.fbs Control table definition

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()


class ControlMode(object):
    Balanced = 0
    Prompt = 1
    Control = 2


class ControlInputType(object):
    Unspecified = 0
    Custom = 1
    Depth = 2
    Canny = 3
    Scribble = 4
    Pose = 5
    Normalbae = 6
    Color = 7
    Lineart = 8
    Softedge = 9
    Seg = 10
    Inpaint = 11
    Ip2p = 12
    Shuffle = 13
    Mlsd = 14
    Tile = 15
    Blur = 16
    Lowquality = 17
    Gray = 18


class Control(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Control()
        x.Init(buf, n + offset)
        return x

    def Init(self, buf, pos):
        self._tab = flatbuffers.Table(buf, pos)

    # Slot 0: file (string)
    def File(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Slot 1: weight (float, default 1)
    def Weight(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 1.0

    # Slot 2: guidance_start (float, default 0)
    def GuidanceStart(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # Slot 3: guidance_end (float, default 1)
    def GuidanceEnd(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 1.0

    # Slot 4: no_prompt (bool, default false)
    def NoPrompt(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # Slot 5: global_average_pooling (bool, default true)
    def GlobalAveragePooling(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return True

    # Slot 6: down_sampling_rate (float, default 1)
    def DownSamplingRate(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 1.0

    # Slot 7: control_mode (ControlMode byte, default Balanced)
    def ControlModeVal(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return ControlMode.Balanced

    # Slot 8: target_blocks ([string])
    def TargetBlocks(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4)
        return None

    def TargetBlocksLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Slot 9: input_override (ControlInputType byte, default Unspecified)
    def InputOverride(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return ControlInputType.Unspecified


# --- Control Builder Functions ---

def ControlStart(builder):
    builder.StartObject(10)  # 10 fields (slots 0-9)

def Start(builder):
    ControlStart(builder)

def ControlEnd(builder):
    return builder.EndObject()

def End(builder):
    return ControlEnd(builder)

def ControlAddFile(builder, file):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(file), 0)

def AddFile(builder, file):
    ControlAddFile(builder, file)

def ControlAddWeight(builder, weight):
    builder.PrependFloat32Slot(1, weight, 1.0)

def AddWeight(builder, weight):
    ControlAddWeight(builder, weight)

def ControlAddGuidanceStart(builder, guidanceStart):
    builder.PrependFloat32Slot(2, guidanceStart, 0.0)

def AddGuidanceStart(builder, guidanceStart):
    ControlAddGuidanceStart(builder, guidanceStart)

def ControlAddGuidanceEnd(builder, guidanceEnd):
    builder.PrependFloat32Slot(3, guidanceEnd, 1.0)

def AddGuidanceEnd(builder, guidanceEnd):
    ControlAddGuidanceEnd(builder, guidanceEnd)

def ControlAddNoPrompt(builder, noPrompt):
    builder.PrependBoolSlot(4, noPrompt, 0)

def AddNoPrompt(builder, noPrompt):
    ControlAddNoPrompt(builder, noPrompt)

def ControlAddGlobalAveragePooling(builder, globalAveragePooling):
    builder.PrependBoolSlot(5, globalAveragePooling, 1)

def AddGlobalAveragePooling(builder, globalAveragePooling):
    ControlAddGlobalAveragePooling(builder, globalAveragePooling)

def ControlAddDownSamplingRate(builder, downSamplingRate):
    builder.PrependFloat32Slot(6, downSamplingRate, 1.0)

def AddDownSamplingRate(builder, downSamplingRate):
    ControlAddDownSamplingRate(builder, downSamplingRate)

def ControlAddControlMode(builder, controlMode):
    builder.PrependUint8Slot(7, controlMode, 0)

def AddControlMode(builder, controlMode):
    ControlAddControlMode(builder, controlMode)

def ControlStartTargetBlocksVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartTargetBlocksVector(builder, numElems):
    return ControlStartTargetBlocksVector(builder, numElems)

def ControlAddTargetBlocks(builder, targetBlocks):
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(targetBlocks), 0)

def AddTargetBlocks(builder, targetBlocks):
    ControlAddTargetBlocks(builder, targetBlocks)

def ControlAddInputOverride(builder, inputOverride):
    builder.PrependUint8Slot(9, inputOverride, 0)

def AddInputOverride(builder, inputOverride):
    ControlAddInputOverride(builder, inputOverride)
