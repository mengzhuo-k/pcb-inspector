"""Candidate localization should reject simple silkscreen and empty printed rings."""
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from solder_sites import detect_sites


class SolderCandidateTests(unittest.TestCase):
    def fixture(self,path,scale=1):
        picture=np.full((900,1200,3),(45,105,30),dtype=np.uint8)
        cv2.putText(picture,'R12',(550,220),cv2.FONT_HERSHEY_SIMPLEX,1.3,(225,225,225),2)
        cv2.circle(picture,(750,450),35,(230,230,230),3)
        cv2.rectangle(picture,(700,580),(800,630),(230,230,230),3)
        cv2.rectangle(picture,(202,180),(310,245),(25,25,25),-1)
        for left in (150,320):
            gradient=np.tile(np.linspace(120,245,45,dtype=np.uint8),(60,1))
            picture[180:240,left:left+45]=gradient[:,:,None]
        if scale!=1:picture=cv2.resize(picture,None,fx=scale,fy=scale)
        cv2.imencode('.png',picture)[1].tofile(str(path))

    def test_textured_terminals_are_kept_but_printing_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'board.png';self.fixture(path)
            sites=detect_sites(path)
        self.assertEqual(len(sites),2)
        self.assertTrue(all(site['x2']<.4 for site in sites))
        self.assertTrue(all(0<=site['x1']<site['x2']<=1 and 0<=site['y1']<site['y2']<=1 for site in sites))
        self.assertEqual(len({site['id'] for site in sites}),len(sites))

    def test_localization_is_consistent_for_resized_board(self):
        with tempfile.TemporaryDirectory() as folder:
            full=Path(folder)/'full.png';small=Path(folder)/'small.png'
            self.fixture(full);self.fixture(small,.5)
            first,second=detect_sites(full),detect_sites(small)
        self.assertEqual(len(first),len(second))
        for a,b in zip(first,second):
            for key in ('x1','y1','x2','y2'):self.assertAlmostEqual(a[key],b[key],delta=.01)

    def test_only_plated_holes_and_opposing_smd_pads_are_returned(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'targets.png';self.fixture(path)
            picture=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
            cv2.circle(picture,(500,450),16,(205,205,205),-1)
            cv2.circle(picture,(500,450),10,(20,20,20),-1)
            # An isolated, reflective rectangle is not the terminal pair requested.
            gradient=np.tile(np.linspace(120,245,45,dtype=np.uint8),(60,1))
            picture[700:760,500:545]=gradient[:,:,None]
            cv2.imencode('.png',picture)[1].tofile(str(path))
            sites=detect_sites(path)
        kinds=[site['kind'] for site in sites]
        self.assertEqual(kinds.count('through_hole'),1)
        self.assertEqual(kinds.count('smd_joint'),2)
        self.assertEqual(set(kinds),{'through_hole','smd_joint'})
        self.assertFalse(any(site['x1']<.44 and site['x2']>.41 and site['y1']>.7 for site in sites))

    def test_opposing_rectangular_pads_do_not_require_a_mounted_component(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'empty-footprint.png'
            picture=np.full((900,1200,3),(45,105,30),dtype=np.uint8)
            for left in (150,230):
                gradient=np.tile(np.linspace(120,245,30,dtype=np.uint8),(80,1))
                picture[200:280,left:left+30]=gradient[:,:,None]
            cv2.imencode('.png',picture)[1].tofile(str(path))
            sites=detect_sites(path)
        self.assertEqual(len(sites),2)
        self.assertEqual({site['kind'] for site in sites},{'smd_joint'})

    def test_invalid_precision_and_non_board_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'blank.png'
            cv2.imencode('.png',np.full((300,300,3),255,dtype=np.uint8))[1].tofile(str(path))
            with self.assertRaises(ValueError):detect_sites(path,'unknown')
            with self.assertRaises(ValueError):detect_sites(path)
